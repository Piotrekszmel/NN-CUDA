#pragma once

struct Shape {
    size_t x;
    size_t y;

    Shape(size_t size_x = 1, size_t size_y = 1) {
        x = size_x;
        y = size_y;
    }
};
