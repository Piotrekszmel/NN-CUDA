#pragma once

#include <iostream>

struct Shape {
    size_t x;
    size_t y;

    Shape(size_t x = 1, size_t y = 1);
    friend std::ostream& operator<< (std::ostream& o, const Shape& shape);
};
