package main

import (
	"fmt"
	"github.com/runningwild/go-fftw/fftw"
)

func main() {
	fmt.Printf("hello, world\n")
	d10 := fftw.NewArray(10)

	for i := 0; i < 10; i++ {
		d10.Set(i, complex(float64(i), float64(0)))
	}

	fmt.Printf("d10:\n")
	printArray(d10)

	fmt.Printf("d10hat:\n")
	d10hat := fftw.FFT(d10)
	printArray(d10hat)

	fmt.Printf("d10hatinv:\n")
	d10hatinv := fftw.IFFT(d10hat)
	constTimesArray(d10hatinv, 1.0/10.0)

	printArray(d10hatinv)

	Ny, Nx := 4, 8
	arr := fftw.NewArray2(Ny, Nx)

	for i := 0; i < arr.N[0]; i++ {
		for j := 0; j < arr.N[1]; j++ {
			arr.Set(i, j, complex(float64(i), float64(j)))
		}
	}

	fmt.Printf("2D Array \n")
	printArray2(arr)

	// physical parameters
	//g := 9.81*0.1
	//f0 := 1.0e-4
	//cd := 0.0025
	Lx := 5.0e3
	Ly := 5.0e3

	dx := Lx / float64(Nx)
	dy := Ly / float64(Ny)

	fmt.Printf("dx: %f, dy: %f\n", dx, dy)

	x := range1D(1, Nx)
	constTimesArray(x, complex(dx, 0.0))
	fmt.Printf("x: \n")
	printArray(x)

	y := range1D(1, Ny)
	constTimesArray(y, complex(dy, 0.0))
	fmt.Printf("y: \n")
	printArray(y)

	yy := repeatColumns(y, Nx)
	fmt.Printf("yy: \n")
	printArray2(yy)

	xx := repeatRows(x, Ny)
	fmt.Printf("xx: \n")
	printArray2(xx)

	PI := 3.14159265358979323
	dk := 2.0*PI / Lx
	dl := 2.0*PI / Ly

	fmt.Printf("dk: %f\n", dk)
	fmt.Printf("dl: %f\n", dl)

    k := buildWaveNumber(Nx, Lx)
    fmt.Printf("k:\n")
    printArray(k)

    l := buildWaveNumber(Ny, Ly)
	fmt.Printf("l:\n")
	printArray(l)

    kk := repeatRows(k, Ny)
    fmt.Printf("kk:\n")
    printArray2(kk)

    ll := repeatColumns(l, Nx)
    fmt.Printf("ll:\n")
    printArray2(ll)


}

// Allocates new memory for wavenumber array.
func buildWaveNumber(N int, L float64) *fftw.Array {
    PI := 3.14159265358979323
    dk := 2.0*PI / L

    k := fftw.NewArray(N)
    k.Set(0, 0.0)
    for i := 1; i <= N/2-1; i++ {
        k.Set(i, complex(float64(i), 0.0))
        k.Set(N/2+i, complex(float64(-N/2+i), 0.0))
    }
    k.Set(N/2, complex(float64(N/2), 0.0))
    constTimesArray(k, complex(dk, 0.0))

    return k

}

func printArray(a *fftw.Array) {
	for i := 0; i < a.Len(); i++ {
		fmt.Printf("%v\n", a.At(i))
	}
}

// scales by constant in-place.
func constTimesArray(a *fftw.Array, c complex128) {
	for i := 0; i < a.Len(); i++ {
		a.Set(i, c*a.At(i))
	}
}

func printArray2(a *fftw.Array2) {
	for i := 0; i < a.N[0]; i++ {
		for j := 0; j < a.N[1]; j++ {
			fmt.Printf("%v ", a.At(i, j))
		}
		fmt.Printf("\n")
	}
}

// Allocates new memory for return array.
func range1D(start int, end int) *fftw.Array {
	a := fftw.NewArray(end - start + 1)
	for i := 0; i <= end-start; i++ {
		a.Set(i, complex(float64(i+start), 0.0))
	}

	return a
}

func repeatColumns(input *fftw.Array, repeats int) *fftw.Array2 {
	length := input.Len()
	a := fftw.NewArray2(length, repeats)
	for j := 0; j < repeats; j++ {
		for i := 0; i < length; i++ {
			a.Set(i, j, input.At(i))
		}
	}

	return a
}

func repeatRows(input *fftw.Array, repeats int) *fftw.Array2 {
	length := input.Len()
	a := fftw.NewArray2(repeats, length)
	for i := 0; i < repeats; i++ {
		for j := 0; j < length; j++ {
			a.Set(i, j, input.At(j))
		}
	}

	return a
}
