#include <stdlib.h>
#include <stdio.h>

#include "lib.h"

int main() {
    l_ComputeNode* leaf_a = l_new_leaf(8.8, false);
    l_ComputeNode* leaf_b = l_new_leaf(3.3, true);
    l_ComputeNode* leaf_c = l_new_leaf(2.0, false);
    l_ComputeNode* addition = l_new_addition(leaf_a, leaf_b);
    l_ComputeNode* multiplication = l_new_multiplication(addition, leaf_c);

    printf("%f, %d\n", l_evaluate(leaf_a), l_requires_grad(leaf_a));
    printf("%f, %d\n", l_evaluate(leaf_b), l_requires_grad(leaf_b));
    printf("%f, %d\n", l_evaluate(leaf_c), l_requires_grad(leaf_c));
    printf("%f, %d\n", l_evaluate(addition), l_requires_grad(addition));
    printf("%f, %d\n", l_evaluate(multiplication), l_requires_grad(multiplication));

    l_delete_leaf(leaf_a);
    l_delete_leaf(leaf_b);
    l_delete_operation(addition);
    l_delete_operation(multiplication);
    return 0;
}
