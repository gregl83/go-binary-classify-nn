package log

import (
	"os"
	"bytes"
	"strings"
	"io"
	"fmt"
	"time"
	"sync"
	"sort"
	"encoding/json"

	"golang.org/x/crypto/ssh/terminal"

	"github.com/sirupsen/logrus"
)

const (
	defaultTimestampFormat = "2006-01-02 15:04:05"
	// Log coloring
	nocolor = 0
	red     = 31
	green   = 32
	yellow  = 33
	blue    = 36
	gray    = 37
)

var (
	baseTimestamp time.Time
)

func init() {
	baseTimestamp = time.Now()
}

// This is to not silently overwrite `time`, `msg` and `level` fields when
// dumping it. If this code wasn't there doing:
//
//  logrus.WithField("level", 1).Info("hello")
//
// Would just silently drop the user provided level. Instead with this code
// it'll logged as:
//
//  {"level": "info", "fields.level": 1, "msg": "hello", "time": "..."}
//
// It's not exported because it's still using Data in an opinionated way. It's to
// avoid code duplication between the two default formatters.
func prefixFieldClashes(data logrus.Fields) {
	if t, ok := data["time"]; ok {
		data["fields.time"] = t
	}

	if m, ok := data["msg"]; ok {
		data["fields.msg"] = m
	}

	if l, ok := data["level"]; ok {
		data["fields.level"] = l
	}
}

// LineFormatter formats logs into line format use by the monolog php package
type LineFormatter struct {
	// Set to true to bypass checking for a TTY before outputting colors.
	ForceColors bool

	// Force disabling colors.
	DisableColors bool

	// Disable timestamp logging. useful when output is redirected to logging
	// system that already adds timestamps.
	DisableTimestamp bool

	// Enable logging the full timestamp when a TTY is attached instead of just
	// the time passed since beginning of execution.
	FullTimestamp bool

	// TimestampFormat to use for display when a full timestamp is printed
	TimestampFormat string

	// The fields are sorted by default for a consistent output. For applications
	// that log extremely frequently and don't use the JSON formatter this may not
	// be desired.
	DisableSorting bool

	// QuoteEmptyFields will wrap empty fields in quotes if true
	QuoteEmptyFields bool

	// Whether the logger's out is to a terminal
	isTerminal bool

	sync.Once
}

func (f *LineFormatter) init(entry *logrus.Entry) {
	if entry.Logger != nil {
		f.isTerminal = f.checkIfTerminal(entry.Logger.Out)
	}
}

func (f *LineFormatter) checkIfTerminal(w io.Writer) bool {
	switch v := w.(type) {
	case *os.File:
		return terminal.IsTerminal(int(v.Fd()))
	default:
		return false
	}
}


// Format renders a single line log entry
func (f *LineFormatter) Format(entry *logrus.Entry) ([]byte, error) {
	var b *bytes.Buffer
	keys := make([]string, 0, len(entry.Data))
	for k := range entry.Data {
		keys = append(keys, k)
	}

	if !f.DisableSorting {
		sort.Strings(keys)
	}
	if entry.Buffer != nil {
		b = entry.Buffer
	} else {
		b = &bytes.Buffer{}
	}

	prefixFieldClashes(entry.Data)

	f.Do(func() { f.init(entry) })

	isColored := (f.ForceColors || f.isTerminal) && !f.DisableColors

	timestampFormat := f.TimestampFormat
	if timestampFormat == "" {
		timestampFormat = defaultTimestampFormat
	}
	if isColored {
		f.printColored(b, entry, keys, timestampFormat)
	} else {
		f.printStandard(b, entry, keys, timestampFormat)
	}

	b.WriteByte('\n')
	return b.Bytes(), nil
}

func (f *LineFormatter) printStandard(b *bytes.Buffer, entry *logrus.Entry, keys []string, timestampFormat string) {
	levelText := strings.ToUpper(entry.Level.String())

	if f.DisableTimestamp {
		fmt.Fprintf(b, "main.%s: %s", levelText, entry.Message)
	} else if !f.FullTimestamp {
		fmt.Fprintf(b, "[%04d] main.%s: %s", int(entry.Time.Sub(baseTimestamp)/time.Second), levelText, entry.Message)
	} else {
		fmt.Fprintf(b, "[%s] main.%s: %s", entry.Time.Format(timestampFormat), levelText, entry.Message)
	}

	if len(keys) > 0 {
		logContext := new(bytes.Buffer)
		json.NewEncoder(logContext).Encode(entry.Data)
		fmt.Fprintf(b, " %s", strings.TrimSpace(logContext.String()))
	}
}

func (f *LineFormatter) printColored(b *bytes.Buffer, entry *logrus.Entry, keys []string, timestampFormat string) {
	var levelColor int
	switch entry.Level {
	case logrus.DebugLevel:
		levelColor = gray
	case logrus.WarnLevel:
		levelColor = yellow
	case logrus.ErrorLevel, logrus.FatalLevel, logrus.PanicLevel:
		levelColor = red
	default:
		levelColor = blue
	}

	levelText := strings.ToUpper(entry.Level.String())

	if f.DisableTimestamp {
		fmt.Fprintf(b, "\x1b[%dm main.%s: %s\x1b[0m", levelColor, levelText, entry.Message)
	} else if !f.FullTimestamp {
		fmt.Fprintf(b, "\x1b[%dm[%04d] main.%s: %s\x1b[0m", levelColor, int(entry.Time.Sub(baseTimestamp)/time.Second), levelText, entry.Message)
	} else {
		fmt.Fprintf(b, "\x1b[%dm[%s] main.%s: %s\x1b[0m", levelColor, entry.Time.Format(timestampFormat), levelText, entry.Message)
	}

	if len(keys) > 0 {
		logContext := new(bytes.Buffer)
		json.NewEncoder(logContext).Encode(entry.Data)
		fmt.Fprintf(b, "\x1b[%dm %s\x1b[0m", levelColor, strings.TrimSpace(logContext.String()))
	}
}