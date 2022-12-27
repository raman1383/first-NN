#include <iostream>
using namespace std;

int partition(int array[], int start, int end)
{
    int pivot = array[start];

    int count = 0;
    for (int i = start + 1; i <= end; i++)
        if (array[i] <= pivot)
            count++;

    int pivot_index = start + count;
    swap(array[pivot_index], array[start]);

    int i = start, j = end;

    while (i < pivot_index && j > pivot_index)
    {
        while (array[i] <= pivot)
            i++;

        while (array[j] > pivot)
            j--;

        if (i < pivot_index && j > pivot_index)
            swap(array[i++], array[j--]);
    }

    return pivot;
}

void quick_sort(int arr[], int start, int end)
{
    if (start >= end)
        return;

    int p = partition(arr, start, end);
    quick_sort(arr, start, p - 1);
    quick_sort(arr, p + 1, end);
}

int main()
{
    int arr[] = {9, 3, 4, 2, 1, 8};
    int n = 6;

    quick_sort(arr, 0, n - 1);

    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";

    return 0;
}