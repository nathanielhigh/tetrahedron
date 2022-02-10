#include "app.hpp"

#include <functional>
#include <memory>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <optional>

using namespace std;
using namespace glm;

struct MVP {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

void tetrahedron(vector<vec3>& vertices, vector<int>& indices) {
	vertices = {
		normalize(vec3(1, 1, 1)),
		normalize(vec3(1, -1, -1)),
		normalize(vec3(-1, 1, -1)),
		normalize(vec3(-1, -1, 1))
	};

	indices = {
		0, 1, 2,
		1, 3, 2,
		3, 0, 2,
		0, 3, 1
	};
}

// divide triangle by finding center of triangle and midpoints of legs
void divide3(vector<vec3>& vertices, vector<int>& indices, bool norm = false)
{
	vector<int> indexes;

	map<pair<int, int>, int> v;

	auto getIndex = [&](int i, int j) {
		if (!v.contains(minmax(indices[i], indices[j])))
		{
			vertices.push_back((vertices[indices[i]] + vertices[indices[j]]) / 2.f);
			int index = vertices.size() - 1;
			v[minmax(indices[i], indices[j])] = index;
			return index;
		}
		return v[minmax(indices[i], indices[j])];
	};

	for (int i = 0; i < indices.size(); i += 3) {

		int ab = getIndex(i, i + 1);
		int ac = getIndex(i, i + 2);
		int bc = getIndex(i + 1, i + 2);

		int a = indices[i];
		int b = indices[i + 1];
		int c = indices[i + 2];

		vec3 m = (vertices[a] + vertices[b] + vertices[c]) / 3.f;

		if (norm) {
			vertices[ab] = normalize(vertices[ab]);
			vertices[ac] = normalize(vertices[ac]);
			vertices[bc] = normalize(vertices[bc]);

			vertices[a] = normalize(vertices[a]);
			vertices[b] = normalize(vertices[b]);
			vertices[c] = normalize(vertices[c]);

			m = normalize(m);
		}

		vertices.push_back(m);

		int mi = vertices.size() - 1;

		indexes.insert(indexes.end(), {
				a, ab, mi,
				ab, b, mi,
				mi, b, bc,
				mi, bc, c,
				ac, mi, c,
				a, mi, ac
			});
	}

	indices = indexes;
}

// divide triangle by finding center of triangle
void divide2(vector<vec3>& vertices, vector<int>& indices, bool norm = false)
{
	vector<int> indexes;

	for (int i = 0; i < indices.size(); i += 3) {

		int a = indices[i];
		int b = indices[i + 1];
		int c = indices[i + 2];


		if (norm) {

			vertices[a] = normalize(vertices[a]);
			vertices[b] = normalize(vertices[b]);
			vertices[c] = normalize(vertices[c]);

			//m = normalize(m);
		}

		vec3 m = (vertices[a] + vertices[b] + vertices[c]) / 3.f;


		vertices.push_back(m);

		int mi = vertices.size() - 1;

		indexes.insert(indexes.end(), {
				a, b, mi,
				b, c, mi,
				a, mi, c
			});
	}
	indices = indexes;
}

// divide triangle by finding midpoints of each leg
void divide(vector<vec3>& vertices, vector<int>& indices, bool norm = false)
{
	vector<int> indexes;

	map<pair<int, int>, int> v;

	auto getIndex = [&](int i, int j) {
		if (!v.contains(minmax(indices[i], indices[j])))
		{
			vertices.push_back((vertices[indices[i]] + vertices[indices[j]]) / 2.f);
			int index = vertices.size() - 1;
			v[minmax(indices[i], indices[j])] = index;
			return index;
		}
		return v[minmax(indices[i], indices[j])];
	};

	for (int i = 0; i < indices.size(); i += 3) {

		int ab = getIndex(i, i + 1);
		int ac = getIndex(i, i + 2);
		int bc = getIndex(i + 1, i + 2);

		int a = indices[i];
		int b = indices[i + 1];
		int c = indices[i + 2];

		if (norm) {
			vertices[ab] = normalize(vertices[ab]);
			vertices[ac] = normalize(vertices[ac]);
			vertices[bc] = normalize(vertices[bc]);

			vertices[a] = normalize(vertices[a]);
			vertices[b] = normalize(vertices[b]);
			vertices[c] = normalize(vertices[c]);
		}

		indexes.insert(indexes.end(), {
				a, ab, ac,
				ab, bc, ac,
				ab, b, bc,
				bc, c, ac
			});
	}
	indices = indexes;
}

function divideFunc = divide;

vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	size_t fileSize = file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

void resizeCallback(GLFWwindow* window, int width, int height)
{
	reinterpret_cast<App*>(glfwGetWindowUserPointer(window))->resized();
}

void App::resized()
{
	cout << "window resized" << endl;
}

App::App()
{
	try
	{
		tetrahedron(vertices, indices);

		createWindow();
		createInstance();
		pickPhysicalDevice();
		createSurface();
		createDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}
	catch (exception e)
	{
		cout << e.what() << endl;
	}
}

void App::createWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	window = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, resizeCallback);
}

void App::createInstance()
{
	vk::ApplicationInfo appInfo("Vulkan", VK_MAKE_VERSION(1, 0, 0), "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_0);

	unsigned int glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	vector<const char*> extensions;
	for (int i = 0; i < glfwExtensionCount; ++i) {
		extensions.push_back(glfwExtensions[i]);
	}

	//extensions.push_back("VK_KHR_get_physical_device_properties2");

	vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
	vk::InstanceCreateInfo createInfo(vk::InstanceCreateFlags(), &appInfo, validationLayers, extensions);
	instance = make_shared<vk::raii::Instance>(context.createInstance(createInfo));
}

void App::pickPhysicalDevice()
{
	physicalDevice = make_shared<vk::raii::PhysicalDevice>(move(instance->enumeratePhysicalDevices().front()));
}

void App::createSurface()
{
	VkSurfaceKHR surf;
	glfwCreateWindowSurface(**instance, window, nullptr, &surf);
	surface = make_shared<vk::raii::SurfaceKHR>(vk::raii::SurfaceKHR(*instance, surf));
}

void App::createDevice()
{
	auto props = physicalDevice->getQueueFamilyProperties();

	graphicsIndex = 0;
	presentIndex = 0;

	for (int i = 0; i < props.size(); ++i)
	{
		if (props[i].queueFlags & vk::QueueFlagBits::eGraphics) {
			graphicsIndex = i;
		}
		if (graphicsIndex == i && physicalDevice->getSurfaceSupportKHR(i, **surface)) {
			presentIndex = i;
		}
	}

	float priority = 1.f;

	vector<vk::DeviceQueueCreateInfo> queueInfos = { vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), graphicsIndex, 1, &priority) };

	std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	vk::DeviceCreateInfo info(vk::DeviceCreateFlags(), queueInfos, {}, deviceExtensions, {});

	device = make_shared<vk::raii::Device>(physicalDevice->createDevice(info));

	graphicsQueue = make_shared<vk::raii::Queue>(device->getQueue(graphicsIndex, 0));
	presentQueue = make_shared<vk::raii::Queue>(device->getQueue(presentIndex, 0));

}

void App::createSwapChain()
{
	auto capabilities = physicalDevice->getSurfaceCapabilitiesKHR(**surface);
	auto formats = physicalDevice->getSurfaceFormatsKHR(**surface);
	auto modes = physicalDevice->getSurfacePresentModesKHR(**surface);

	auto format = [=] {
		for (const auto& form : formats)
		{
			if (form.format == vk::Format::eB8G8R8A8Srgb && form.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
			{
				return form;
			}
		}
		return formats[0];
	}();

	auto extent = [=] {
		if (capabilities.currentExtent.width != UINT32_MAX)
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			vk::Extent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

			actualExtent.width = std::max(capabilities.minImageExtent.width,
				std::min(capabilities.maxImageExtent.width, actualExtent.width));

			actualExtent.height = std::max(capabilities.minImageExtent.height,
				std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}();

	auto mode = [=] {
		for (const auto& md : modes)
		{
			if (md == vk::PresentModeKHR::eMailbox)
			{
				return md;
			}
		}
		return vk::PresentModeKHR::eFifo;
	}();

	vk::SwapchainCreateInfoKHR info(
		vk::SwapchainCreateFlagsKHR(),
		**surface,
		capabilities.minImageCount + 1,
		format.format,
		format.colorSpace,
		extent,
		1,
		vk::ImageUsageFlagBits::eColorAttachment,
		vk::SharingMode::eExclusive,
		{},
		capabilities.currentTransform,
		vk::CompositeAlphaFlagBitsKHR::eOpaque,
		mode,
		true);

	swapChain = make_shared<vk::raii::SwapchainKHR>(device->createSwapchainKHR(info));

	swapChainImageFormat = format;
	swapChainExtent = extent;
}

void App::createImageViews()
{
	for (auto& image : swapChain->getImages())
	{
		vk::ImageViewCreateInfo info(
			vk::ImageViewCreateFlags(),
			image,
			vk::ImageViewType::e2D,
			swapChainImageFormat.format,
			vk::ComponentMapping(),
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		swapChainImageViews.push_back(make_shared<vk::raii::ImageView>(device->createImageView(info)));
	}
}

void App::createRenderPass()
{
	vector<vk::AttachmentDescription> attachments = {
		vk::AttachmentDescription(
			vk::AttachmentDescriptionFlags(),
			swapChainImageFormat.format,
			vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare,
			vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::ePresentSrcKHR)
	};

	std::vector<vk::AttachmentReference> reference = { vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal) };

	vector<vk::SubpassDescription> subpass = {
		vk::SubpassDescription(
			vk::SubpassDescriptionFlags(),
			vk::PipelineBindPoint::eGraphics,
			{},
			reference
		)
	};

	vk::RenderPassCreateInfo info(vk::RenderPassCreateFlags(), attachments, subpass);

	renderPass = make_shared<vk::raii::RenderPass>(device->createRenderPass(info));
}

void App::createDescriptorSetLayout()
{
	vector<vk::DescriptorSetLayoutBinding> binding = { vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex) };
	vk::DescriptorSetLayoutCreateInfo info(vk::DescriptorSetLayoutCreateFlags(), binding);
	descriptorLayout = make_shared<vk::raii::DescriptorSetLayout>(device->createDescriptorSetLayout(info));
}

void App::createGraphicsPipeline()
{
	auto vertShaderCode = readFile("vert.spv");
	auto geomShaderCode = readFile("geom.spv");
	auto fragShaderCode = readFile("frag.spv");

	vk::ShaderModuleCreateInfo vertInfo(vk::ShaderModuleCreateFlags(), vertShaderCode.size(),
		reinterpret_cast<uint32_t*>(vertShaderCode.data()));
	vk::ShaderModuleCreateInfo geomInfo(vk::ShaderModuleCreateFlags(), geomShaderCode.size(),
		reinterpret_cast<uint32_t*>(geomShaderCode.data()));
	vk::ShaderModuleCreateInfo fragInfo(vk::ShaderModuleCreateFlags(), fragShaderCode.size(),
		reinterpret_cast<uint32_t*>(fragShaderCode.data()));

	auto vertShaderModule = device->createShaderModule(vertInfo);
	auto geomShaderModule = device->createShaderModule(geomInfo);
	auto fragShaderModule = device->createShaderModule(fragInfo);

	vector<vk::PipelineShaderStageCreateInfo> stageInfo = {
		vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "main"),
		vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main")
	};

	vector<vk::VertexInputBindingDescription> binding = { vk::VertexInputBindingDescription(0, sizeof(vertices[0])) };
	vector < vk::VertexInputAttributeDescription> attribute = { vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0) };

	vk::PipelineVertexInputStateCreateInfo vertexInfo(vk::PipelineVertexInputStateCreateFlags(), binding, attribute);

	vk::PipelineInputAssemblyStateCreateInfo assemblyInfo(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList);

	vector<vk::Viewport> viewport = { vk::Viewport(0.f, 0.f, swapChainExtent.width, swapChainExtent.height, 0.f, 1.f) };
	vector<vk::Rect2D> scissor = { vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent) };

	vk::PipelineViewportStateCreateInfo viewportInfo(vk::PipelineViewportStateCreateFlags(), viewport, scissor);

	vk::PipelineRasterizationStateCreateInfo rasterizationInfo(
		vk::PipelineRasterizationStateCreateFlags(), false, false,
		vk::PolygonMode::eLine, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise,
		false, 0.f, 0.f, 0.f, 1.f
	);

	vk::PipelineMultisampleStateCreateInfo multisampleInfo(vk::PipelineMultisampleStateCreateFlags(),
		vk::SampleCountFlagBits::e1, false, 1.f, nullptr, false, false);

	vector<vk::PipelineColorBlendAttachmentState> attachment = {
		vk::PipelineColorBlendAttachmentState(
			false,
			vk::BlendFactor::eOne,
			vk::BlendFactor::eZero,
			vk::BlendOp::eAdd,
			vk::BlendFactor::eOne,
			vk::BlendFactor::eZero,
			vk::BlendOp::eAdd,
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
	};

	array<float, 4> blendConstants = { 0.f, 0.f, 0.f, 0.f };

	vk::PipelineColorBlendStateCreateInfo colorInfo(vk::PipelineColorBlendStateCreateFlags(),
		false, vk::LogicOp::eCopy, attachment, blendConstants);

	vector<vk::DescriptorSetLayout> descriptorInfo = { **descriptorLayout };

	vk::PipelineLayoutCreateInfo pipelineInfo(vk::PipelineLayoutCreateFlags(), descriptorInfo);

	pipelineLayout = make_shared<vk::raii::PipelineLayout>(device->createPipelineLayout(pipelineInfo));

	vk::GraphicsPipelineCreateInfo graphicsInfo(
		vk::PipelineCreateFlags(),
		stageInfo,
		&vertexInfo,
		&assemblyInfo,
		nullptr,
		&viewportInfo,
		&rasterizationInfo,
		&multisampleInfo,
		nullptr,
		&colorInfo,
		nullptr,
		**pipelineLayout,
		**renderPass
	);

	pipeline = make_shared<vk::raii::Pipeline>(device->createGraphicsPipeline(nullptr, graphicsInfo));
}

void App::createFramebuffers()
{
	for (int i = 0; i < swapChainImageViews.size(); ++i) {
		vector<vk::ImageView> attachments = { **swapChainImageViews[i] };
		vk::FramebufferCreateInfo info(vk::FramebufferCreateFlags(), **renderPass, attachments, swapChainExtent.width, swapChainExtent.height, 1);
		framebuffers.push_back(make_shared<vk::raii::Framebuffer>(device->createFramebuffer(info)));
	}
}

void App::createCommandPool()
{
	vk::CommandPoolCreateInfo info(vk::CommandPoolCreateFlags(), graphicsIndex);
	pool = make_shared<vk::raii::CommandPool>(device->createCommandPool(info));
}

void App::createVertexBuffer()
{
	vk::BufferCreateInfo info(vk::BufferCreateFlags(), sizeof(vertices[0]) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer);
	vertexBuffer = make_shared<vk::raii::Buffer>(device->createBuffer(info));

	auto memReq = vertexBuffer->getMemoryRequirements();
	auto memProp = physicalDevice->getMemoryProperties();

	auto prop = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

	auto index = [=] {
		for (int i = 0; i < memProp.memoryTypeCount; ++i) {
			if ((memReq.memoryTypeBits & (1 << i)) &&
				memProp.memoryTypes[i].propertyFlags & prop)
			{
				return i;
			}
		}
		return 0;
	}();

	vk::MemoryAllocateInfo allocInfo(memReq.size, index);

	vertexMemory = make_shared<vk::raii::DeviceMemory>(device->allocateMemory(allocInfo));

	vertexBuffer->bindMemory(**vertexMemory, 0);

	void* mem = vertexMemory->mapMemory(0, info.size);
	memcpy(mem, vertices.data(), info.size);
	vertexMemory->unmapMemory();
}

void App::createIndexBuffer()
{
	vk::BufferCreateInfo info(vk::BufferCreateFlags(), sizeof(indices[0]) * indices.size(), vk::BufferUsageFlagBits::eIndexBuffer);
	indexBuffer = make_shared<vk::raii::Buffer>(device->createBuffer(info));

	auto memReq = indexBuffer->getMemoryRequirements();
	auto memProp = physicalDevice->getMemoryProperties();

	auto prop = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

	auto index = [=] {
		for (int i = 0; i < memProp.memoryTypeCount; ++i) {
			if ((memReq.memoryTypeBits & (1 << i)) &&
				memProp.memoryTypes[i].propertyFlags & prop)
			{
				return i;
			}
		}
		return 0;
	}();

	vk::MemoryAllocateInfo allocInfo(memReq.size, index);

	indexMemory = make_shared<vk::raii::DeviceMemory>(device->allocateMemory(allocInfo));

	indexBuffer->bindMemory(**indexMemory, 0);

	void* mem = indexMemory->mapMemory(0, info.size);
	memcpy(mem, indices.data(), info.size);
	indexMemory->unmapMemory();
}

void App::createUniformBuffers()
{
	for (int i = 0; i < swapChain->getImages().size(); ++i) {
		vk::BufferCreateInfo info(vk::BufferCreateFlags(), sizeof(MVP), vk::BufferUsageFlagBits::eUniformBuffer);
		uniformBuffers.push_back(make_shared<vk::raii::Buffer>(device->createBuffer(info)));

		auto memReq = uniformBuffers[i]->getMemoryRequirements();
		auto memProp = physicalDevice->getMemoryProperties();

		auto prop = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

		auto index = [=] {
			for (int i = 0; i < memProp.memoryTypeCount; ++i) {
				if ((memReq.memoryTypeBits & (1 << i)) &&
					memProp.memoryTypes[i].propertyFlags & prop)
				{
					return i;
				}
			}
			return 0;
		}();

		vk::MemoryAllocateInfo allocInfo(memReq.size, index);

		uniformMemory.push_back(make_shared<vk::raii::DeviceMemory>(device->allocateMemory(allocInfo)));

		uniformBuffers[i]->bindMemory(**uniformMemory[i], 0);
	}
}

void App::createDescriptorPool()
{
	vector<vk::DescriptorPoolSize> sizes = { vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, swapChain->getImages().size()) };
	vk::DescriptorPoolCreateInfo info(vk::DescriptorPoolCreateFlags(), swapChain->getImages().size(), sizes);

	descriptorPool = make_shared<vk::raii::DescriptorPool>(device->createDescriptorPool(info));
}

void App::createDescriptorSets()
{
	vector<vk::DescriptorSetLayout> layout(swapChain->getImages().size(), **descriptorLayout);

	vk::DescriptorSetAllocateInfo info(**descriptorPool, layout);

	for (auto& set : device->allocateDescriptorSets(info))
	{
		descriptorSet.push_back(make_shared<vk::raii::DescriptorSet>(move(set)));
	}

	for (int i = 0; i < swapChain->getImages().size(); ++i)
	{
		std::vector<vk::DescriptorBufferInfo> bufferInfo = { vk::DescriptorBufferInfo(**uniformBuffers[i], 0, sizeof(MVP)) };
		std::vector<vk::WriteDescriptorSet> writes = { vk::WriteDescriptorSet(**descriptorSet[i], 0, 0, vk::DescriptorType::eUniformBuffer, nullptr, bufferInfo) };
		device->updateDescriptorSets(writes, nullptr);
	}
}

void App::createCommandBuffers()
{
	vk::CommandBufferAllocateInfo allocateInfo(**pool, vk::CommandBufferLevel::ePrimary, framebuffers.size());
	for (auto& buffer : device->allocateCommandBuffers(allocateInfo)) {
		commandBuffers.push_back(make_shared<vk::raii::CommandBuffer>(move(buffer)));
	}

	for (int i = 0; i < commandBuffers.size(); ++i) {

		vk::CommandBufferBeginInfo bufferInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
		commandBuffers[i]->begin(bufferInfo);

		array<float, 4> values = { 0.f, 0.f, 0.f, 0.f };

		std::vector<vk::ClearValue> clear = { vk::ClearValue(vk::ClearColorValue(values)) };

		vk::RenderPassBeginInfo renderInfo(
			**renderPass,
			**framebuffers[i],
			vk::Rect2D({ 0, 0 }, swapChainExtent),
			clear
		);

		commandBuffers[i]->beginRenderPass(renderInfo, vk::SubpassContents::eInline);
		commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);
		commandBuffers[i]->bindVertexBuffers(0, { **vertexBuffer }, { 0 });
		commandBuffers[i]->bindIndexBuffer(**indexBuffer, 0, vk::IndexType::eUint32);

		commandBuffers[i]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipelineLayout, 0, { **descriptorSet[i] }, nullptr);

		commandBuffers[i]->drawIndexed(indices.size(), 1, 0, 0, 0);
		commandBuffers[i]->endRenderPass();
		commandBuffers[i]->end();

	}
}

void App::createSyncObjects()
{
	vk::SemaphoreCreateInfo semaphoreInfo;
	vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
	imagesInFlight.resize(swapChainImageViews.size());

	for (int i = 0; i < max_frames; ++i) {
		imageSemaphors.push_back(make_shared<vk::raii::Semaphore>(device->createSemaphore(semaphoreInfo)));
		renderSemaphors.push_back(make_shared<vk::raii::Semaphore>(device->createSemaphore(semaphoreInfo)));
		fences.push_back(make_shared<vk::raii::Fence>(device->createFence(fenceInfo)));
	}
}

void App::drawFrame()
{
	updateControls();

	device->waitForFences({ **fences[frame] }, true, UINT64_MAX);
	vk::AcquireNextImageInfoKHR info(**swapChain, UINT64_MAX, **imageSemaphors[frame]);
	auto result = swapChain->acquireNextImage(UINT64_MAX, **imageSemaphors[frame]);
	uint32_t imageIndex = result.second;


	if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
		device->waitForFences({ **imagesInFlight[imageIndex] }, true, UINT64_MAX);
	}

	imagesInFlight[imageIndex] = fences[frame];
	device->resetFences({ **fences[frame] });
	updateUniforms(imageIndex);

	vector<vk::Semaphore> waitSemaphores = { **imageSemaphors[frame] };
	vector<vk::PipelineStageFlags> waitStages = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	vector<vk::CommandBuffer> buffers = { **commandBuffers[imageIndex] };
	vector<vk::Semaphore> signalSemaphores = { **renderSemaphors[frame] };

	vk::SubmitInfo submitInfo(waitSemaphores, waitStages, buffers, signalSemaphores);
	graphicsQueue->submit(submitInfo, **fences[frame]);
	vector<vk::SwapchainKHR> swapChains = { **swapChain };
	vector<uint32_t> indices = { imageIndex };

	vk::PresentInfoKHR presentInfo(signalSemaphores, swapChains, indices);
	presentQueue->presentKHR(presentInfo);
	frame = (frame + 1) % max_frames;
}

void App::updateUniforms(int imageIndex)
{
	static auto startTime = chrono::high_resolution_clock::now();
	auto currentTime = chrono::high_resolution_clock::now();
	float time = chrono::duration<float>(startTime - currentTime).count();

	MVP mvp;
	mvp.model = rotate(mat4(1.f), time * pi<float>() / 2, vec3(0.f, 0.f, 1.f));
	mvp.view = lookAt(vec3(0.f, 5.f, 1.f), vec3(0.f, 0.f, 0.f), vec3(0.f, 0.f, 1.f));
	mvp.proj = perspective(pi<float>() / 4.f, static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.f);
	mvp.proj[1][1] *= -1;

	void* mem = uniformMemory[imageIndex]->mapMemory(0, sizeof(MVP));
	memcpy(mem, &mvp, sizeof(MVP));
	uniformMemory[imageIndex]->unmapMemory();
}

void App::updateControls()
{
	static bool down = false;

	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS && !down)
	{
		divideFunc(vertices, indices, glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);
		commandBuffers.clear();
		createVertexBuffer();
		createIndexBuffer();
		createCommandBuffers();
		down = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS && !down)
	{
		divideFunc = divide;
		down = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS && !down)
	{
		divideFunc = divide2;
		down = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS && !down)
	{
		divideFunc = divide3;
		down = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !down)
	{
		tetrahedron(vertices, indices);
		commandBuffers.clear();
		createVertexBuffer();
		createIndexBuffer();
		createCommandBuffers();
		down = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_RELEASE && down)
	{
		down = false;
	}
}