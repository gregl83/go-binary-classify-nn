package adapters

import (
	"database/sql"
	"time"
)

const dateFormat = "2006-01-02"

// store adapter implementing Store interface
type store struct {
	db *sql.DB
}

// CreateLog will create a new log record and return ID
func (s *store) CreateLog(logDate time.Time) int64 {
	stmt, err := s.db.Prepare("INSERT INTO currency_rates_log(date) VALUES(?)")
	if err != nil {
		return 0
	}

	res, err := stmt.Exec(logDate.Format(dateFormat))
	if err != nil {
		return 0
	}

	id, err := res.LastInsertId()
	if err != nil {
		return 0
	}

	return id
}

// CompleteLog will update a log record w/completed time and return error if unsuccessful
func (s *store) CompleteLog(id int64) error {
	stmt, err := s.db.Prepare("UPDATE currency_rates_log SET completed = ? WHERE id = ?")
	if err != nil {
		return nil
	}

	_, err = stmt.Exec(time.Now(), id)
	if err != nil {
		return nil
	}

	return nil
}

// ReadLastLog will read the and return last log record for a given date
func (s *store) ReadLastLog(logDate time.Time) (*Log, error) {
	var (
		id int
		created time.Time
		completed *time.Time
	)

	stmt, err := s.db.Prepare("SELECT id, created, completed FROM currency_rates_log WHERE date = ? ORDER BY date, created DESC LIMIT 1")
	if err != nil {
		return nil, err
	}

	err = stmt.QueryRow(logDate.Format(dateFormat)).Scan(&id, &created, &completed)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		} else {
			return nil, err
		}
	}

	if completed == nil {
		completed = &time.Time{}
	}

	log := Log{
		id,
		logDate,
		created,
		*completed,
	}

	return &log, nil
}

// ReadCurrencies from store
func (s *store) ReadCurrencies() ([]Currency, error) {
	var (
		currencies []Currency
		id int
		alphabeticCode string
		numericCode int
	)

	rows, err := s.db.Query("SELECT id, alphabetic_code, numeric_code FROM currencies")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		if err := rows.Scan(&id, &alphabeticCode, &numericCode); err != nil {
			return nil, err
		}
		currencies = append(
			currencies,
			Currency{
				ID: id,
				AlphabeticCode: alphabeticCode,
				NumericCode: numericCode,
			},
		)
	}

	if err = rows.Err(); err != nil {
		return nil, err
	}

	return currencies, nil
}

// CreateRates in store
func (s *store) CreateRates(currencyRates []CurrencyRate) error {
	var values []interface{}

	query := "INSERT IGNORE INTO currency_rates(base_currency_numeric_code, currency_numeric_code, rate, date) VALUES"
	for _, currencyRate := range currencyRates {
		query += "(?, ?, ?, ?),"
		values = append(
			values,
			currencyRate.BaseCurrency.NumericCode,
			currencyRate.Currency.NumericCode,
			currencyRate.Rate,
			currencyRate.Date.Format(dateFormat),
		)
	}
	// remove trailing comma
	query = query[0:len(query)-1]

	stmt, err := s.db.Prepare(query)
	if err != nil {
		return err
	}

	_, err = stmt.Exec(values...)
	if err != nil {
		return err
	}

	return nil
}

// NewStore adapter
func NewStore(db *sql.DB) *store {
	return &store{db: db}
}