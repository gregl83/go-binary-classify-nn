// Package log implements logging functionality with helpers
package log

import (
	"github.com/sirupsen/logrus"
)

var (
	// Logger is a logrus instance with line formatter
	Logger *logrus.Logger
)

func init() {
	Logger = logrus.New()

	Logger.Formatter = &LineFormatter{
		FullTimestamp: true,
	}

	Logger.SetLevel(logrus.DebugLevel)
}

// FailOnError logs a fatal when err isn't nil (dies)
func FailOnError(err error, message string) {
	if err != nil {
		Logger.Fatalf("%s: %s", message, err)
	}
}

// FailOnEmptyString logs fatal when value is "" (dies)
func FailOnEmptyString(val string, message string) {
	if val == "" {
		Logger.Fatalf("%s", message)
	}
}