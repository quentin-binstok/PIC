# Parameters

[Table of contents](toc.md)

The available parameters can be grouped in several ways.

## General parameters

### `grid`

- Type: array of 2 ints
- Role: defines the size of the rectangular grid. Is the number of nodes.

### `log_file`

- Type: string
- Role: is the name of the log file
- Default: `"log.txt"`

## Boundary and initial conditions

### `bc`

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

### `ic_vx`, `ic_vy` and `ic_p`

- Type: array of objects
- Role: defines the initial conditions
- Default: `value` of 0 if unspecified

`ic_vx` and `ic_vy` define the initial velocity, and `ic_p` defines the initial pressure.

The objects are to be as:

```json
{
	"tl": [0, 0],
	"br": [5, 5],
	"value": 1.4
}
```

The `tl` parameter stands for "top-left", and the `br` stands for "bottom-right". They define the extend on which the initial condition applies. The `value` is a float.
