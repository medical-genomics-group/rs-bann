# rs-bann

## Requirements

### ArrayFire

This package uses ![ArrayFire](https://github.com/arrayfire/arrayfire-rust),
which needs to be installed in order to be able to compile.

This can be done on Ubuntu by first ![adding the package](https://github.com/arrayfire/arrayfire/wiki/Install-ArrayFire-From-Linux-Package-Managers#ubuntu), restarting the shell and then installing via apt:

```console
sudo apt install arrayfire
```

Then, ![add the path to the lib files to env variables](https://github.com/arrayfire/arrayfire-rust#use-from-cratesio--). If installed with `apt`, the root installation dir should be `/usr`.

Then, restart the shell and potentially run `cargo clean`.

### Further requirements

Further requirements are

- cmake
- gfortran

which can be innstalled on Ubuntu with:

```console
sudo apt install cmake gfortran
```

### Known issues

#### Flaky tests

If `cargo test` is flaky, try a different device: `AF_CUDA_DEFAULT_DEVICE=1 cargo test`
