#pragma once

template <typename T, int N>
class vec
{
private:
	T _data[N];
	
	template <int M, typename ... Args>
	static void unroll(T *ptr, T arg, Args ... args)
	{
		static_assert(M < N, "too much args");
		*ptr = arg;
		unroll<M + 1>(ptr + 1, args ...);
	}
	
	template <int M>
	static void unroll(T *ptr, T arg)
	{
		static_assert(M == N, "args count mismatch");
		*ptr = arg;
	}
	
public:
	vec() = default;
	template <typename ... Args>
	vec(Args ... args)
	{
		unroll<1>(_data, args ...);
	}
	
	T *data()
	{
		return _data;
	}
	
	const T *data() const
	{
		return _data;
	}
};

typedef vec<float, 2> fvec2;
typedef vec<float, 3> fvec3;
typedef vec<float, 4> fvec4;
typedef vec<int, 2> ivec2;
typedef vec<int, 3> ivec3;
typedef vec<int, 4> ivec4;
