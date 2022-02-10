#include "app.hpp"

int main()
{
	App app;
	while (!glfwWindowShouldClose(app.window))
	{
		glfwPollEvents();
		app.drawFrame();
	}

	app.wait();
}
