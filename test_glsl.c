#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

int main() {
    glfwInit();
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(640, 480, "Test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    
    const char* src = "#version 430\n"
                      "uniform sampler2D MainImage;\n"
                      "uniform sampler2D MainImage;\n"
                      "out vec4 color;\n"
                      "void main() { color = texture(MainImage, vec2(0.5)); }\n";
    
    GLuint shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        printf("Error: %s\n", infoLog);
    } else {
        printf("Success!\n");
    }
    return 0;
}
