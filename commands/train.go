package commands

import (
	"github.com/spf13/cobra"

	"github.com/gregl83/go-binary-classify-nn/infrastructure/log"
)

// trainCmd represents the train command
var trainCmd = &cobra.Command{
	Use:   "train",
	Short: "Train neural network",
	Long: "Train the neural network using a data source",
	Run: func(cmd *cobra.Command, args []string) {
		log.Logger.Info("training neural network")

		// todo - load data

		// todo - run train

		// todo - persist trained weights

		log.Logger.Info("training completed")
	},
}

func init() {
	rootCmd.AddCommand(trainCmd)
}