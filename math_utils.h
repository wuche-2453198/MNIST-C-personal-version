#pragma once


template <typename T>
T square(T number)
{
    return number * number;
}

// 平方差
template <typename T>
T squared_difference(T num1, T num2)
{
    return square(num1 - num2);
}