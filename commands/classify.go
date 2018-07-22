package commands

import (
	"github.com/spf13/cobra"

	"github.com/gregl83/go-binary-classify-nn/infrastructure/log"
)

// classifyCmd represents the classify command
var classifyCmd = &cobra.Command{
	Use:   "classify",
	Short: "Classify input",
	Long: "Classify input using trained neural network",
	Run: func(cmd *cobra.Command, args []string) {
		log.Logger.Info("classifying input")

		// todo - train

		log.Logger.Info("classification completed")
	},
}

func init() {
	rootCmd.AddCommand(classifyCmd)
}