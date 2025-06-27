/**
 * Assignment: Design a Class Hierarchy with Polymorphism
 *
 * Define a base class `Shape` with a pure virtual method `area()`, and derive classes like `Circle`
 * and `Rectangle` that override `area()` to provide appropriate calculations.
 * Demonstrate runtime polymorphism by calling `area()` through a base class pointer or reference.
 * Ensure the base class has a virtual destructor to allow proper cleanup of derived objects.
 *
 * Bonus:
 * - Add an interface `Printable` with a method `print()`.
 * - Have derived classes implement both `Shape` and `Printable` to show multiple inheritance.
 * - Optionally, discuss extensibility and the Open/Closed Principle.
 */

#include <cmath>
#include <iostream>
#include <memory>

// Interface class for printable objects
class Printable {
  public:
    virtual void print() const = 0;
    virtual ~Printable() = default;
};

// Abstract base class
class Shape {
  public:
    virtual double area() const = 0;
    virtual ~Shape() { std::cout << "Shape destroyed\n"; }
};

// Circle class
class Circle : public Shape, public Printable {
  private:
    double radius;

  public:
    explicit Circle(double r) : radius(r) {}
    double area() const override { return M_PI * radius * radius; }
    void print() const override { std::cout << "Circle with radius " << radius << ", area = " << area() << "\n"; }
    ~Circle() override { std::cout << "Circle destroyed\n"; }
};

// Rectangle class
class Rectangle : public Shape, public Printable {
  private:
    double width, height;

  public:
    explicit Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override { return width * height; }
    void print() const override {
        std::cout << "Rectangle with width " << width << " and height " << height << ", area = " << area() << "\n";
    }
    ~Rectangle() override { std::cout << "Rectangle destroyed\n"; }
};

int main() {
    std::unique_ptr<Shape> s1 = std::make_unique<Circle>(5.0);
    std::unique_ptr<Shape> s2 = std::make_unique<Rectangle>(4.0, 6.0);

    std::cout << "s1 area: " << s1->area() << "\n";
    std::cout << "s2 area: " << s2->area() << "\n";

    Printable *p1 = dynamic_cast<Printable *>(s1.get());
    Printable *p2 = dynamic_cast<Printable *>(s2.get());

    if (p1)
        p1->print();
    if (p2)
        p2->print();

    return 0;
}

/*
s1 area: 78.5398
s2 area: 24
Circle with radius 5, area = 78.5398
Rectangle with width 4 and height 6, area = 24
Rectangle destroyed
Shape destroyed
Circle destroyed
Shape destroyed
*/
