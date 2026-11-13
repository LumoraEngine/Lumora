#include "renderer/RenderGraph.hpp"

#include "core/JobSystem.hpp"
#include "renderer/GpuProfiler.hpp"
#include "renderer/TransientImagePool.hpp"
#include "renderer/vulkan/VulkanCommon.hpp"

#include <algorithm>
#include <utility>

namespace lumora::render {

// ---- FramebufferCache --------------------------------------------------------------------------

nvrhi::IFramebuffer* FramebufferCache::get(nvrhi::IDevice* device, const nvrhi::FramebufferDesc& desc, std::uint64_t frame) {
    std::vector<const void*> key;
    for (const auto& a : desc.colorAttachments) key.push_back(a.texture);
    key.push_back(desc.depthAttachment.texture);
    key.push_back(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(desc.depthAttachment.isReadOnly ? 1 : 0)));
    Entry& entry = entries_[key];
    if (!entry.framebuffer) {
        entry.framebuffer = device->createFramebuffer(desc);
        if (!entry.framebuffer) throw rhi::VulkanError("could not create a framebuffer", VK_ERROR_INITIALIZATION_FAILED);
    }
    entry.lastUsed = frame;
    return entry.framebuffer;
}

void FramebufferCache::prune(std::uint64_t frame) {
    std::erase_if(entries_, [frame](const auto& e) { return e.second.lastUsed + 8 < frame; });
}

// ---- PassBuilder ---------------------------------------------------------------------------------

void PassBuilder::writeColor(RGImage image, LoadOp load, glm::vec4 clear) {
    graph_.addAccess(pass_, {image, ImageUsage::ColorAttachment, load, clear});
}

void PassBuilder::writeDepth(RGImage image, LoadOp load, float clearDepth) {
    graph_.addAccess(pass_, {image, ImageUsage::DepthAttachment, load, glm::vec4(clearDepth)});
}

void PassBuilder::readDepth(RGImage image) { graph_.addAccess(pass_, {image, ImageUsage::DepthReadOnly}); }
void PassBuilder::sample(RGImage image) { graph_.addAccess(pass_, {image, ImageUsage::Sampled}); }
void PassBuilder::writeStorage(RGImage image) { graph_.addAccess(pass_, {image, ImageUsage::Storage}); }
void PassBuilder::copyFrom(RGImage image) { graph_.addAccess(pass_, {image, ImageUsage::CopySource}); }
void PassBuilder::copyTo(RGImage image) { graph_.addAccess(pass_, {image, ImageUsage::CopyDestination, LoadOp::DontCare}); }
void PassBuilder::recordInParallel(bool parallel) { graph_.passes_.at(pass_).parallel = parallel; }
void PassBuilder::keepAlways() { graph_.passes_.at(pass_).keep = true; }

// ---- PassContext ---------------------------------------------------------------------------------

nvrhi::ITexture* PassContext::texture(RGImage h) const { return graph_.resource(h).physical.texture; }
std::uint32_t PassContext::bindlessIndex(RGImage h) const { return graph_.resource(h).physical.bindlessIndex; }

nvrhi::ViewportState PassContext::viewport() const {
    const nvrhi::FramebufferInfoEx& info = framebuffer->getFramebufferInfo();
    return nvrhi::ViewportState().addViewportAndScissorRect(
        nvrhi::Viewport(static_cast<float>(info.width), static_cast<float>(info.height)));
}

void PassContext::prepareSampled(nvrhi::ICommandList* list) const {
    // Bindless reads are invisible to NVRHI's tracking: declare them.
    bool any = false;
    for (const RenderGraph::Access& a : graph_.passes_[pass_].accesses) {
        if (a.usage != ImageUsage::Sampled) continue;
        list->setTextureState(graph_.resource(a.image).physical.texture, nvrhi::AllSubresources,
                              nvrhi::ResourceStates::ShaderResource);
        any = true;
    }
    if (any) list->commitBarriers();
}

void PassContext::recordParallel(std::size_t count, std::size_t grain, const ChunkFn& fn) {
    lastChunks_ = 0;
    if (count == 0) return;
    const RenderGraph::Pass& pass = graph_.passes_[pass_];
    grain = std::max<std::size_t>(grain, 1);
    const std::size_t chunks = std::min<std::size_t>((count + grain - 1) / grain, runtime_.maxThreads);
    if (!pass.parallel || !runtime_.allowParallel || !runtime_.newList || chunks < 2) {
        fn(cmd, 0, count);
        return;
    }
    // The pass so far ends this list; the chunks follow in their own lists, then a new list
    // continues with whatever the pass and the next passes record.
    cmd->close();
    std::vector<nvrhi::CommandListHandle> lists(chunks);
    for (auto& list : lists) {
        list = runtime_.newList();
        prepareSampled(list);
    }
    const std::size_t per = (count + chunks - 1) / chunks;
    JobSystem::global().parallelFor(chunks, 1, [&](std::size_t begin, std::size_t end) {
        for (std::size_t c = begin; c < end; ++c) {
            const std::size_t first = c * per;
            const std::size_t last = std::min(count, first + per);
            if (first < last) fn(lists[c], first, last);
        }
    });
    for (auto& list : lists) {
        list->close();
        graph_.lists_.push_back(list);
    }
    nvrhi::CommandListHandle next = runtime_.newList();
    graph_.lists_.push_back(next);
    cmd = next;
    lastChunks_ = static_cast<std::uint32_t>(chunks);
    graph_.parallelChunks_ += lastChunks_;
}

// ---- RenderGraph ---------------------------------------------------------------------------------

RGImage RenderGraph::importImage(std::string name, const ImportedImage& image) {
    Resource r;
    r.name = std::move(name);
    r.imported = true;
    r.physical = image;
    if (image.texture) r.desc = {image.texture->getDesc().format, image.texture->getDesc().width, image.texture->getDesc().height};
    resources_.push_back(std::move(r));
    return {static_cast<std::uint32_t>(resources_.size() - 1)};
}

RGImage RenderGraph::createImage(std::string name, const TransientImageDesc& desc) {
    Resource r;
    r.name = std::move(name);
    r.desc = desc;
    resources_.push_back(std::move(r));
    return {static_cast<std::uint32_t>(resources_.size() - 1)};
}

void RenderGraph::addRasterPass(std::string name, const SetupFn& setup, ExecuteFn execute) {
    passes_.push_back({std::move(name), true, false, false, {}, std::move(execute)});
    PassBuilder builder(*this, static_cast<std::uint32_t>(passes_.size() - 1));
    setup(builder);
}

void RenderGraph::addPass(std::string name, const SetupFn& setup, ExecuteFn execute) {
    passes_.push_back({std::move(name), false, false, false, {}, std::move(execute)});
    PassBuilder builder(*this, static_cast<std::uint32_t>(passes_.size() - 1));
    setup(builder);
}

void RenderGraph::addComputePass(std::string name, const SetupFn& setup, ExecuteFn execute) {
    passes_.push_back({std::move(name), false, false, true, {}, std::move(execute)});
    PassBuilder builder(*this, static_cast<std::uint32_t>(passes_.size() - 1));
    setup(builder);
    std::uint32_t storage = 0;
    for (const Access& a : passes_.back().accesses) storage += a.usage == ImageUsage::Storage ? 1 : 0;
    if (storage > kStorageSlots) {
        throw rhi::VulkanError("render graph: compute pass '" + passes_.back().name + "' has more storage images than slots",
                               VK_ERROR_UNKNOWN);
    }
}

void RenderGraph::addAccess(std::uint32_t pass, const Access& access) {
    if (!access.image.valid() || access.image.id >= resources_.size()) {
        throw rhi::VulkanError("render graph: pass '" + passes_[pass].name + "' uses an invalid image handle",
                               VK_ERROR_UNKNOWN);
    }
    passes_[pass].accesses.push_back(access);
}

const RenderGraph::Resource& RenderGraph::resource(RGImage h) const { return resources_.at(h.id); }

namespace {

bool writes(ImageUsage usage) {
    return usage == ImageUsage::ColorAttachment || usage == ImageUsage::DepthAttachment || usage == ImageUsage::Storage ||
           usage == ImageUsage::CopyDestination;
}

std::uint64_t imageBytes(const TransientImageDesc& d) { return render::imageBytes(d.format, d.width, d.height, d.depth); }

} // namespace

void RenderGraph::compile(bool shareTextures) {
    // What earlier passes wrote matters only to the passes that read it: a colour or depth
    // attachment that is loaded (blended into, depth-tested), or an image sampled, copied or
    // used as storage. Backwards from the last pass, a pass runs when someone needs its output.
    auto reads = [](const Access& a) {
        switch (a.usage) {
            case ImageUsage::ColorAttachment:
            case ImageUsage::DepthAttachment:
                return a.load == LoadOp::Load;
            case ImageUsage::CopyDestination:
                return false;
            default:
                return true;
        }
    };
    std::vector<bool> needed(resources_.size(), false);
    for (std::size_t p = passes_.size(); p-- > 0;) {
        Pass& pass = passes_[p];
        bool writesImage = false;
        bool useful = pass.keep;
        for (const Access& a : pass.accesses) {
            if (!writes(a.usage)) continue;
            writesImage = true;
            useful = useful || resources_[a.image.id].imported || needed[a.image.id];
        }
        pass.runs = useful || !writesImage;
        if (!pass.runs) continue;
        for (const Access& a : pass.accesses) {
            if (reads(a)) needed[a.image.id] = true;
        }
    }

    // Lifetimes over the passes that run.
    for (Resource& r : resources_) {
        r.first = UINT32_MAX;
        r.last = 0;
        r.slot = UINT32_MAX;
    }
    for (std::uint32_t p = 0; p < passes_.size(); ++p) {
        if (!passes_[p].runs) continue;
        for (const Access& a : passes_[p].accesses) {
            Resource& r = resources_[a.image.id];
            r.first = std::min(r.first, p);
            r.last = std::max(r.last, p);
        }
    }

    // Textures: in order of first use, an image takes a texture of its size and format that no
    // image alive with it holds (preferably one with the same storage use), else a new one.
    // A texture any of its images writes as storage is made for storage (2D ones are render
    // targets too).
    slots_.clear();
    std::vector<std::uint32_t> order;
    for (std::uint32_t i = 0; i < resources_.size(); ++i) {
        if (!resources_[i].imported && resources_[i].first != UINT32_MAX) order.push_back(i);
    }
    std::ranges::stable_sort(order, {}, [&](std::uint32_t i) { return resources_[i].first; });
    for (std::uint32_t i : order) {
        Resource& r = resources_[i];
        const TransientImageDesc& d = r.desc;
        std::uint32_t chosen = UINT32_MAX;
        if (shareTextures) {
            for (std::uint32_t s = 0; s < slots_.size(); ++s) {
                const TransientImageDesc& t = slots_[s].desc;
                if (slots_[s].last >= r.first || t.format != d.format || t.width != d.width || t.height != d.height ||
                    t.depth != d.depth) {
                    continue;
                }
                if (chosen == UINT32_MAX || (t.storage == d.storage && slots_[chosen].desc.storage != d.storage)) chosen = s;
            }
        }
        if (chosen == UINT32_MAX) {
            chosen = static_cast<std::uint32_t>(slots_.size());
            slots_.push_back({d, 0, i});
        }
        Slot& slot = slots_[chosen];
        slot.desc.storage = slot.desc.storage || d.storage;
        slot.last = r.last;
        r.slot = chosen;
    }
}

std::uint32_t RenderGraph::culledCount() const noexcept {
    return static_cast<std::uint32_t>(std::ranges::count(passes_, false, &Pass::runs));
}

RenderGraph::TransientStats RenderGraph::transientStats() const {
    TransientStats s;
    s.textures = static_cast<std::uint32_t>(slots_.size());
    for (const Slot& slot : slots_) s.bytes += imageBytes(slot.desc);
    for (const Resource& r : resources_) {
        if (r.imported || r.slot == UINT32_MAX) continue;
        ++s.images;
        s.unsharedBytes += imageBytes(r.desc);
    }
    return s;
}

std::vector<nvrhi::CommandListHandle> RenderGraph::execute(nvrhi::CommandListHandle first, GraphRuntime& runtime) {
    parallelChunks_ = 0;
    lists_.clear();
    lists_.push_back(first);
    compile(runtime.shareTransients);
    // A texture from the pool per slot; the images in it take turns.
    std::vector<const PooledImage*> textures(slots_.size());
    for (std::size_t s = 0; s < slots_.size(); ++s) {
        const TransientImageDesc& d = slots_[s].desc;
        textures[s] = &runtime.pool->acquire(d.format, d.width, d.height, resources_[slots_[s].firstImage].name.c_str(), d.storage,
                                             d.depth);
    }
    for (Resource& r : resources_) {
        if (r.imported) continue;
        r.physical = {};
        if (r.slot == UINT32_MAX) continue;
        r.physical.texture = textures[r.slot]->texture;
        r.physical.bindlessIndex = textures[r.slot]->bindlessIndex;
    }
    for (std::uint32_t p = 0; p < passes_.size(); ++p) {
        Pass& pass = passes_[p];
        if (!pass.runs) continue;
        nvrhi::ICommandList* cmd = lists_.back();
        const std::uint32_t scope = runtime.profiler != nullptr ? runtime.profiler->beginScope(cmd, pass.name) : UINT32_MAX;
        PassContext ctx(*this, runtime, p);
        if (pass.raster) {
            nvrhi::FramebufferDesc fb;
            for (const Access& a : pass.accesses) {
                nvrhi::ITexture* texture = resource(a.image).physical.texture;
                switch (a.usage) {
                    case ImageUsage::ColorAttachment:
                        fb.addColorAttachment(texture);
                        if (a.load == LoadOp::Clear) {
                            cmd->clearTextureFloat(texture, nvrhi::AllSubresources,
                                                   nvrhi::Color(a.clear.r, a.clear.g, a.clear.b, a.clear.a));
                        }
                        break;
                    case ImageUsage::DepthAttachment:
                        fb.setDepthAttachment(texture);
                        if (a.load == LoadOp::Clear) {
                            cmd->clearDepthStencilTexture(texture, nvrhi::AllSubresources, true, a.clear.r, false, 0);
                        }
                        break;
                    case ImageUsage::DepthReadOnly:
                        fb.setDepthAttachment(nvrhi::FramebufferAttachment().setTexture(texture).setReadOnly(true));
                        break;
                    default:
                        break;
                }
            }
            // Framebuffers over imported images (the swapchain) are made per frame: those images
            // come and go with their window; the rest are cached.
            const bool imported = std::ranges::any_of(pass.accesses, [&](const Access& a) {
                return resource(a.image).imported && a.usage != ImageUsage::Sampled;
            });
            if (imported) {
                nvrhi::FramebufferHandle handle = runtime.device->createFramebuffer(fb);
                if (!handle) throw rhi::VulkanError("could not create a framebuffer", VK_ERROR_INITIALIZATION_FAILED);
                ownedFramebuffers_.push_back(handle);
                ctx.framebuffer = handle;
            } else {
                ctx.framebuffer = runtime.framebuffers->get(runtime.device, fb, runtime.frame);
            }
        }
        if (pass.compute) {
            // Set 2: the declared storage images in order, empty slots filled with a stand-in.
            nvrhi::BindingSetDesc set;
            std::uint32_t slot = 0;
            for (const Access& a : pass.accesses) {
                if (a.usage != ImageUsage::Storage) continue;
                set.addItem(nvrhi::BindingSetItem::Texture_UAV(slot++, resource(a.image).physical.texture));
            }
            for (; slot < kStorageSlots; ++slot) {
                set.addItem(nvrhi::BindingSetItem::Texture_UAV(slot, runtime.storageFallback));
            }
            nvrhi::BindingSetHandle handle = runtime.device->createBindingSet(set, runtime.storageLayout);
            if (!handle) throw rhi::VulkanError("could not create the storage bindings of '" + pass.name + "'", VK_ERROR_INITIALIZATION_FAILED);
            ownedBindingSets_.push_back(handle);
            ctx.storageSet_ = handle;
        }
        ctx.cmd = cmd;
        ctx.prepareSampled(cmd);
        pass.execute(ctx);
        if (runtime.profiler != nullptr) runtime.profiler->endScope(lists_.back(), scope);
    }
    // Lists replaced by recordParallel were closed there; the caller closes the last one.
    return std::exchange(lists_, {});
}

} // namespace lumora::render
