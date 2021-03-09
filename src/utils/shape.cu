#include "shape.cuh"

Shape::Shape(size_t x, size_t y) {
    this->x = x;
    this->y = y;
}

std::ostream& operator<< (std::ostream& o, const Shape& shape) {
    o << "["<< shape.x << ", "<< shape.y << "]";
    return o;
}

