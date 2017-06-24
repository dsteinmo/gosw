package main

import (
	"bufio"
	"fmt"
	"math"
	"os"

	"github.com/kardianos/osext"
	"github.com/runningwild/go-fftw/fftw"
)

func main() {
	// Grid size.
	Ny, Nx := 128, 128
	N := Nx * Ny

	// physical parameters
	g := 9.81 * 0.1
	f0 := 1.0e-4
	H0 := 20.0
	Lx := 5.0e3
	Ly := 5.0e3
	FINTIME := 1000.0
	CFL := 0.5
	NUMOUTS := 20

	outputInterval := FINTIME / float64(NUMOUTS)

	fmt.Printf("g: %f, f0: %f, H0: %f, Lx: %f, Ly: %f, FINAL TIME: %f, CFL: %f\n", g, f0, H0, Lx, Ly, FINTIME, CFL)

	dx := Lx / float64(Nx)
	dy := Ly / float64(Ny)

	fmt.Printf("dx: %f, dy: %f\n", dx, dy)

	x := constTimesArray(range1D(1, Nx), complex(dx, 0.0))
	y := constTimesArray(range1D(1, Ny), complex(dy, 0.0))

	xx, yy := meshGrid(x, y)

	PI := 3.14159265358979323
	dk := 2.0 * PI / Lx
	dl := 2.0 * PI / Ly

	fmt.Printf("dk: %f\n", dk)
	fmt.Printf("dl: %f\n", dl)

	k := buildWaveNumber(Nx, Lx)
	l := buildWaveNumber(Ny, Ly)

	kk := repeatRows(k, Ny)
	ll := repeatColumns(l, Nx)

	cZero := complex(0.0, 0.0)
	cI := complex(0.0, 1.0)

	ikk := constTimesArray2(kk, cI)

	ill := constTimesArray2(ll, cI)

	c0 := math.Sqrt(g * H0)
	dt := CFL * dx / c0

	NUMSTEPS := FINTIME / dt

	fmt.Printf("c0: %f, dt: %f, NUM STEPS: %f\n", c0, dt, NUMSTEPS)

	eta := fftw.NewArray2(Ny, Nx)
	H := fftw.NewArray2(Ny, Nx)
	u := fftw.NewArray2(Ny, Nx)
	v := fftw.NewArray2(Ny, Nx)
	h := fftw.NewArray2(Ny, Nx)

	// Initialize depth profile, wave elevation, total water column height, and velocity.
	for i := 0; i < Ny; i++ {
		for j := 0; j < Nx; j++ {
			H.Set(i, j, complex(H0, 0.0))
			eta.Set(i, j, complex(0.1*math.Exp(-math.Pow(real((xx.At(i, j)-complex(0.5*Lx, 0.0))/500.0), 2.0)-math.Pow(real((yy.At(i, j)-complex(0.5*Ly, 0.0))/500.0), 2.0)), 0.0))
			h.Set(i, j, H.At(i, j)+eta.At(i, j))
			u.Set(i, j, cZero)
			v.Set(i, j, cZero)
		}
	}

	eta_np1 := fftw.NewArray2(Ny, Nx)
	u_np1 := fftw.NewArray2(Ny, Nx)
	v_np1 := fftw.NewArray2(Ny, Nx)
	h_np1 := fftw.NewArray2(Ny, Nx)

	hu := fftw.NewArray2(Ny, Nx)
	hv := fftw.NewArray2(Ny, Nx)
	hu_x := fftw.NewArray2(Ny, Nx)
	hv_y := fftw.NewArray2(Ny, Nx)

	u_x := fftw.NewArray2(Ny, Nx)
	u_y := fftw.NewArray2(Ny, Nx)
	v_x := fftw.NewArray2(Ny, Nx)
	v_y := fftw.NewArray2(Ny, Nx)

	eta_x := fftw.NewArray2(Ny, Nx)
	eta_y := fftw.NewArray2(Ny, Nx)

	fftscale := complex(1.0/float64(N), 0.0)

	// u_t = -uux -vuy -g*eta_x +fv
	// v_t = -uvx -vvy -g*eta_y -fu
	// eta_t = -(hu)_x - (hv)_y
	t2 := 0.0
	tstep := 0
	for t := 0.0; t < FINTIME; t += dt {
		hu = Array2TimesArray2(h, u)
		hv = Array2TimesArray2(h, v)

		hu_x = constTimesArray2(toRealArray2(fftw.IFFT2(Array2TimesArray2(ikk, fftw.FFT2(hu)))), fftscale)
		hv_y = constTimesArray2(toRealArray2(fftw.IFFT2(Array2TimesArray2(ill, fftw.FFT2(hv)))), fftscale)

		u_x = constTimesArray2(toRealArray2(fftw.IFFT2(Array2TimesArray2(ikk, fftw.FFT2(u)))), fftscale)
		u_y = constTimesArray2(toRealArray2(fftw.IFFT2(Array2TimesArray2(ill, fftw.FFT2(u)))), fftscale)
		v_x = constTimesArray2(toRealArray2(fftw.IFFT2(Array2TimesArray2(ikk, fftw.FFT2(v)))), fftscale)
		v_y = constTimesArray2(toRealArray2(fftw.IFFT2(Array2TimesArray2(ill, fftw.FFT2(v)))), fftscale)

		eta_x = constTimesArray2(toRealArray2(fftw.IFFT2(Array2TimesArray2(ikk, fftw.FFT2(eta)))), fftscale)
		eta_y = constTimesArray2(toRealArray2(fftw.IFFT2(Array2TimesArray2(ill, fftw.FFT2(eta)))), fftscale)

		for i := 0; i < Ny; i++ {
			for j := 0; j < Nx; j++ {
				eta_np1.Set(i, j, eta.At(i, j)-complex(dt, 0.0)*(hu_x.At(i, j)+hv_y.At(i, j)))
				u_np1.Set(i, j, u.At(i, j)-complex(dt, 0.0)*(u.At(i, j)*u_x.At(i, j)+v.At(i, j)*u_y.At(i, j)+complex(g, 0.0)*eta_x.At(i, j)))
				v_np1.Set(i, j, v.At(i, j)-complex(dt, 0.0)*(u.At(i, j)*v_x.At(i, j)+v.At(i, j)*v_y.At(i, j)+complex(g, 0.0)*eta_y.At(i, j)))
				h_np1.Set(i, j, H.At(i, j)+eta_np1.At(i, j))
			}
		}

		eta = eta_np1
		u = u_np1
		v = v_np1
		h = h_np1

		fmt.Printf("t=%f\n", t+dt)

		t2 += dt
		if t2 > outputInterval || t == 0.0 || t+dt >= FINTIME {
			fmt.Printf("outputing.\n")
			outputArray2("h", tstep, h_np1)
			outputArray2("u", tstep, u_np1)
			outputArray2("v", tstep, v_np1)
			t2 = 0.0
		}

		tstep++

	}
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func outputArray2(prefix string, tstep int, a *fftw.Array2) {

	cwd, err := osext.ExecutableFolder()
	check(err)

	fullFileName := fmt.Sprintf("%s/%s%07d.dat", cwd, prefix, tstep)
	f, err := os.Create(fullFileName)
	check(err)
	defer f.Close()

	w := bufio.NewWriter(f)

	for i := 0; i < a.N[0]; i++ {
		for j := 0; j < a.N[1]; j++ {
			_, err = fmt.Fprintf(w, "%1.4f ", real(a.At(i, j)))
			check(err)
		}
		_, err = fmt.Fprintf(w, "\n")
		check(err)
	}

	w.Flush()
}

// Allocates new memory for wavenumber array.
func buildWaveNumber(N int, L float64) *fftw.Array {
	PI := 3.14159265358979323
	dk := 2.0 * PI / L

	k := fftw.NewArray(N)
	k.Set(0, 0.0)
	for i := 1; i <= N/2-1; i++ {
		k.Set(i, complex(float64(i), 0.0))
		k.Set(N/2+i, complex(float64(-N/2+i), 0.0))
	}
	k.Set(N/2, complex(float64(N/2), 0.0))
	k = constTimesArray(k, complex(dk, 0.0))

	return k

}

// scales by constant and returns in newly allocated array.
func constTimesArray(a *fftw.Array, c complex128) *fftw.Array {
	result := fftw.NewArray(a.Len())
	for i := 0; i < a.Len(); i++ {
		result.Set(i, c*a.At(i))
	}

	return result
}

func constTimesArray2(a *fftw.Array2, c complex128) *fftw.Array2 {
	result := fftw.NewArray2(a.N[0], a.N[1])
	for i := 0; i < a.N[0]; i++ {
		for j := 0; j < a.N[1]; j++ {
			result.Set(i, j, c*a.At(i, j))
		}
	}

	return result
}

func copyArray2(a *fftw.Array2) *fftw.Array2 {
	b := fftw.NewArray2(a.N[0], a.N[1])
	for i := 0; i < a.N[0]; i++ {
		for j := 0; j < a.N[1]; j++ {
			b.Set(i, j, a.At(i, j))
		}
	}

	return b
}

func toRealArray2(a *fftw.Array2) *fftw.Array2 {
	b := fftw.NewArray2(a.N[0], a.N[1])
	for i := 0; i < a.N[0]; i++ {
		for j := 0; j < a.N[1]; j++ {
			b.Set(i, j, complex(real(a.At(i, j)), 0.0))
		}
	}

	return b
}

func Array2TimesArray2(x *fftw.Array2, y *fftw.Array2) *fftw.Array2 {
	Ny := x.N[0]
	Nx := x.N[1]

	z := fftw.NewArray2(Ny, Nx)

	for i := 0; i < Ny; i++ {
		for j := 0; j < Nx; j++ {
			z.Set(i, j, x.At(i, j)*y.At(i, j))
		}
	}
	return z
}

func printArray(a *fftw.Array) {
	for i := 0; i < a.Len(); i++ {
		fmt.Printf("%1.4v\n", a.At(i))
	}
}

func printArray2(a *fftw.Array2) {
	for i := 0; i < a.N[0]; i++ {
		for j := 0; j < a.N[1]; j++ {
			fmt.Printf("%1.4f ", real(a.At(i, j)))
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

func meshGrid(x *fftw.Array, y *fftw.Array) (*fftw.Array2, *fftw.Array2) {
	Nx := x.Len()
	Ny := y.Len()
	xx := repeatRows(x, Ny)
	yy := repeatColumns(y, Nx)

	return xx, yy
}
