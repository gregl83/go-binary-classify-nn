# go-binary-classify-nn

GoLang binary classification neural network.

## Usage

`Make install` will build a command line interface utility called `worker`.

Run `worker --help` to see usage instructions.

## Configuration

Commands use the following two golang packages:

- [spf13/cobra](https://github.com/spf13/cobra)
- [spf13/viper](https://github.com/spf13/viper)

Viper is used to handle configurable parameters such as network hyper parameters using env variables.

## Caution

This project has been largely abondoned and will likely get archived. For now, functional deep learning algorithms / functions are covered with tests and available in the [lib](/lib) directory.

## License

MIT
