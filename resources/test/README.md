### small.bed

`small.bed` should correspond to the following genotype matrix in variant-major format.
This matrix has n=20 individuals, m=11 variants.

```python
array([[0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 1],
       [0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1],
       [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
       [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
       [0, 2, 0, 1, 0, 1, 0, 1, 2, 2, 0],
       [0, 0, 0, 1, 0, 2, 1, 1, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2],
       [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
       [0, 1, 0, 0, 0, 1, 1, 2, 1, 1, 1],
       [0, 0, 0, 0, 0, 2, 1, 2, 0, 1, 1],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
       [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
       [0, 1, 0, 0, 0, 1, 0, 1, 2, 1, 0],
       [1, 0, 0, 0, 0, 2, 0, 2, 0, 1, 1],
       [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
       [2, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
       [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]])
```

As a bytearray this should be

```python
bytearray(b'\xef\xbe\xef\xff\xce\xee\xf3\xaa\xbf\xef\xff\xff\xff\xef\xff\xfb\xeb\xef\xef\xaf\xff\xff\xff\xff\xff\xb3\xca\xaa\xbc\xb8\xec\xef\xbf\xfe\xab\xba\xea.\xa8\xa8\xff\xf3\xbf?\xff\xbb\xf2\xaf\xaa\xea\xba\xee\xa3\xfa\xfa')
```

### small.bim 
Accompanying `.bim` file to `small.bed`. 11 markers, located on the first 11 basepairs of chromosome 19.

### small.fam
Accompanying `.fam` file to `small.fam`. 20 individuals, no mothers or fathers in the dataset, all of unknown sex, without phenotype values.