# Handle input as a float or two integers (year, decimals)
from datetime import datetime, timedelta

class DecimalTimeConversion:
    """
    A class to convert decimal year to datetime object.
    """

    @staticmethod
    def convert_dectime_to_date(year, rem=None):
        """
        Convert a decimal year to a datetime object.
        
        :param decimal_year: Decimal year (e.g., 2023.5 or 2023,5) or a tuple of two integers (e.g., (2023, 5))
        :return: Corresponding datetime object
        """
        # Handle input as a single float or two integers
        if rem is None:
            # If only one argument is provided, treat it as a float
            decimal_year = float(year)
            year = int(decimal_year)
            rem = decimal_year - year
        else:
            # If two arguments are provided, calculate rem from the second integer
            rem = rem / 10**len(str(rem))

        # Calculate the number of days in the year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            days_in_year = 366
        else:
            days_in_year = 365
        
        # Calculate the number of seconds in the year and adjust for the decimal part
        seconds_in_year = days_in_year * 24 * 60 * 60
        base = datetime(year, 1, 1)
        date = base + timedelta(seconds=seconds_in_year * rem)
        return date




        # base = datetime(year, 1, 1, tzinfo=CustomUTCOffset())
        # date = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
        # # Format the date to exclude microseconds and UTC offset
        # formatted_date = date.strftime('%Y-%m-%d %H:%M:%S') + CustomUTCOffset.utcoffset(date,timezone.utc)
        # return formatted_date