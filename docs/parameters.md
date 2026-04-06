# Parameters

[Table of contents](toc.md)

The available parameters can be grouped in several ways.

## General parameters

### `solver`

- Type: string
- Role: defines the solver used

This can either be `semi-lagrangian` or `pic`.

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

### `flip`

- Type: float
- Role: defines the fraction of FLIP in a PIC/FLIP simulation
- Default: 0

This parameter is only applicable if the `pic` solver is used. It can be between zero and 1, with 0 being full PIC, and 1 being full FLIP.

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

### `particle_density`

- Type: integer
- Role: the number of particles per cell at the beginning of the simulation
- Default: 8

### `refill`

- Type: bool
- Role: defines whether the liquid cells should be refilled with particles or not
- Default: false

### `rho`

- Type: float
- Role: the density of the fluid

### `gravity`

- Type: bool
- Role: defines whether gravity is to be applied or not
- Default: false

### `g`

- Type: float
- Role: defines the gravitational acceleration
- Default: 9.81

## Boundary and initial conditions

### `bc` - TODO: rewrite this after having changed the code

- Type: array of objects
- Role: defines the boundary conditions
- Default: liquid cells on the boundaries

This parameter defines the boundary conditions to apply to the borders of the domain. It should be given as an array of objects, each being of the form:

```json
{
	"side": 0,
	"type": 2.0
}
```

The `side` parameter defines the side of the domain to which the object will apply. 0 is left, 1 is right, 2 is top, and 3 is down.

The `type` parameter defines the type of cell that will be placed there. 0 is liquid, 1 is solid, 2 is air, and 3 is a forced flow.

If a forced flow is set, then an additional field should be provided. It should either be `speed_x` or `speed_y`, based on which direction is normal to the face. The value should be the speed of the flow at that side.

The code has not been tested either with a type 0 condition, or with a tangential speed applied on a type 3 side. Here be dragons.

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

The possible values can be either `0`, `1`, or `2`. These values correspond to the same thign as the type in the `bc` field.

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
