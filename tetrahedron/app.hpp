#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <vulkan/vulkan_raii.hpp>

#include <iostream>
#include <fstream>
#include <optional>
#include <vector>
#include <memory>
#include <set>

class App
{
public:

	App();

	void resized();
	GLFWwindow* window = nullptr;

	void drawFrame();

	void wait() {
		device->waitIdle();
	}

private:

	std::vector<glm::vec3> vertices;
	std::vector<int> indices;

	void updateUniforms(int imageIndex);
	void updateControls();

	void createWindow();
	void createInstance();
	void pickPhysicalDevice();
	void createSurface();
	void createDevice();
	void createSwapChain();
	void createImageViews();
	void createRenderPass();
	void createDescriptorSetLayout();
	void createGraphicsPipeline();
	void createFramebuffers();
	void createCommandPool();
	void createVertexBuffer();
	void createIndexBuffer();
	void createUniformBuffers();
	void createDescriptorPool();
	void createDescriptorSets();
	void createCommandBuffers();
	void createSyncObjects();


	vk::raii::Context context;
	std::shared_ptr<vk::raii::Instance> instance;
	std::shared_ptr<vk::raii::PhysicalDevice> physicalDevice;
	std::shared_ptr<vk::raii::SurfaceKHR> surface;
	std::shared_ptr<vk::raii::Device> device;
	std::shared_ptr<vk::raii::Queue> graphicsQueue;
	std::shared_ptr<vk::raii::Queue> presentQueue;
	std::shared_ptr<vk::raii::SwapchainKHR> swapChain;
	std::vector<std::shared_ptr<vk::raii::ImageView>> swapChainImageViews;
	std::shared_ptr<vk::raii::RenderPass> renderPass;
	std::shared_ptr<vk::raii::DescriptorSetLayout> descriptorLayout;
	std::shared_ptr<vk::raii::PipelineLayout> pipelineLayout;
	std::shared_ptr<vk::raii::Pipeline> pipeline;
	std::vector<std::shared_ptr<vk::raii::Framebuffer>> framebuffers;
	std::shared_ptr<vk::raii::CommandPool> pool;
	std::vector<std::shared_ptr<vk::raii::CommandBuffer>> commandBuffers;

	std::vector<std::shared_ptr<vk::raii::Semaphore>> imageSemaphors;
	std::vector<std::shared_ptr<vk::raii::Semaphore>> renderSemaphors;
	std::vector<std::shared_ptr<vk::raii::Fence>> fences;
	std::vector<std::shared_ptr<vk::raii::Fence>> imagesInFlight;

	std::shared_ptr<vk::raii::Buffer> vertexBuffer;
	std::shared_ptr<vk::raii::DeviceMemory> vertexMemory;

	std::shared_ptr<vk::raii::Buffer> indexBuffer;
	std::shared_ptr<vk::raii::DeviceMemory> indexMemory;

	std::vector<std::shared_ptr<vk::raii::Buffer>> uniformBuffers;
	std::vector<std::shared_ptr<vk::raii::DeviceMemory>> uniformMemory;

	std::shared_ptr<vk::raii::DescriptorPool> descriptorPool;

	std::vector<std::shared_ptr<vk::raii::DescriptorSet>> descriptorSet;

	vk::SurfaceFormatKHR swapChainImageFormat;
	vk::Extent2D swapChainExtent;
	int graphicsIndex;
	int presentIndex;
	int frame = 0;

	const int max_frames = 2;
};