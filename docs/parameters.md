# Parameters

[Table of contents](toc.md)

The available parameters can be grouped in several ways.

## General parameters

### `solver`

- Type: string
- Role: defines the solver used

This can only be `"semi-lagangian"` for now.

### `iteration_algo`

- Type: string
- Role: defines the iteration algorithm to be used to compute the pressures

This can either be `Jacobi` or `SOR`.

The Jacobi algorithm is parallelized. It is slow to converge.

The successive over-relaxation algorithm (SOR) is a variant of the Gauss-Seidel algorithm that is faster to converge. Instead of replacing the value of the pressure $p_i$ by the newly computed value $p_{new}$, it replaces it by a linear combination of the initial value and the newly computed one. It does $p_{i + 1} = (1 - \omega) p_i + \omega p_{new}$, with $\omega \in [1, 1.95]$. This algorithm is also parallelized, using a red-black variant. This is made possible by the fact that for each cell, only the up, down, left, and right cells are needed for the computation.

### `sampling_rate`

- Type: integer
- Role: defines the frequency at which the data is to be saved. $0$ means that no data should be saved.

### `log_file`

- Type: string
- Role: is the name of the log file
- Default: `"log.txt"`

## Physical & simulation parameters

### `grid`

- Type: array of 2 ints
- Role: defines the size of the rectangular grid. Is the number of nodes.

### `space_steps`

- Type: array of 2 floats
- Role: defines the spatial steps.

### `nt`

- Type: integer
- Role: the number of time steps of the simulation

### `delta_t`

- Type: float
- Role: defines the time step

### `tol`

- Type: float
- Role: the tolerance used to stop the iteration algorithm

### `max_iter`

- Type: integer
- Role: the maximum number of iterations for the iteration algorithm

### `rho`

- Type: float
- Role: the density of the fluid

## Boundary and initial conditions

### `bc` - TODO: rewrite this after having changed the code

- Type: array of objects
- Role: defines the boundary conditions
- Default: `"closed"` if unspecified

The objects that are to be put in the array are to be according to the following model:

```json
{
	"start": [0, 0],
	"end": [2, 0],
	"type": "closed OR open",
	"speed": 0.5,
	"pressure": 6
}
```

The `start` and `end` define the side on which the condition will apply. The `type`, which is to be either `"closed"` or `"open"`, defines if the side is an impermeable surface, or an open channel. If it is open, the `speed` and `pressure` parameters are necessary.

### `ic_vx` and `ic_vy`

- Type: array of objects
- Role: defines the initial conditions
- Default: `value` of 0 if unspecified

`ic_vx` and `ic_vy` define the initial velocity.

The objects are to be as:

```json
{
	"tl": [0, 0],
	"br": [5, 5],
	"value": 1.4
}
```

The `tl` parameter stands for "top-left", and the `br` stands for "bottom-right". They define the extend on which the initial condition applies. The bottom-right index is included in the setting of the values. The `value` is a float.

### `ic_cell`

- Type: array of objects
- Role: defines special cells in the domain
- Default: all cells are liquid

This parameter allows to define special cells in the domain, such as solid cells.

The objects are to be as:

```json
{
	"tl": [20, 20],
	"br": [50, 50],
	"value": 1.0
}
```

The possible values can be either `0` or `1`, the former referring to liquid, and the latter to solid. Other values will for now cause an undefined behaviour.

### `ic_cylinders`

- Type: array of objects
- Role: defines the cylinders present in the domain
- Default: no cylinders present

This parameter allows to define cylinders of solid in the domain.

The objects are to be as:

```json
{
	"center": [20, 50],
	"radius": 10
}
```

## User-defined fields

It is possible to define arbitrary fields, that fill be advected by the solver. To do this, two things must be done: the list of such fields must be provided, and their initial conditions must be given.

The list of fields must be provided with a parameter `"fields"`, which must be an array of strings. Each string must be the name of a field. For example, one might have:

```json
"fields": ["temperature", "concentration"]
```

Afterwards, the initial conditions of the fields must be provided. To do this, add a parameter that is the name of the field, which must be an array of objects. The objects must be as follows:

```json
{
	"tl": [20, 20],
	"br": [50, 50],
	"value": 4.5
}
```

For example, one might have

```json
"temperature": [
	{
		"tl": [200, 235],
		"br": [230, 265],
		"value": 5.0
	},
	{
		"tl": [50, 235],
		"br": [90, 265],
		"value": 5.0
	},
	{
		"tl": [235, 400],
		"br": [265, 450],
		"value": 5.0
	}
],

"concentration": [
	{
		"tl": [30, 235],
		"br": [80, 265],
		"value": 10.0
	}
]
```
