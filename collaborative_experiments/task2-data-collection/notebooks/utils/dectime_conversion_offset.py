
# -*- coding: utf-8 -*-
# __author__ = "Sara Hajbane"

"""
This module provides a utility function to convert decimal year to datetime object. 
It is useful for handling time series data where dates are represented in decimal format.

The CustomUTCOffset is created specifically for the Litter Windrows Dataset to adjust 
reported non datestr times to match UTC to match CodeT for accurate Sentinel product retrieval 
"""

from datetime import datetime, timedelta, tzinfo, timezone

class CustomUTCOffset(tzinfo):
    def utcoffset(self, dt):
        # Offset of 59 minutes and 55 seconds
        return timedelta(hours= 1, minutes=00, seconds=12)

    def dst(self, dt):
        # No daylight saving time adjustment
        return timedelta(0)

    def tzname(self, dt):
        # Return the timezone name as a datetime.timezone object
        return timezone(self.utcoffset(dt))


class DecimalTimeConversionOS:
    """
    A class to convert decimal year to datetime object.
    """

    @staticmethod
    def dectime_to_date(year, rem=None):
        """
        Convert a decimal year to a datetime object.
        
        :param decimal_year: Decimal year as a float (e.g., 2023.5) 
        :or two integers (e.g., 2023,5)
        :return: Corresponding datetime object with litter windrows specific utcoffset
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

        # Calculate the number of days from the decimal part
        # base = datetime(year, 1, 1, tzinfo=CustomUTCOffset())
        # date = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
        base = datetime(year, 1, 1, tzinfo=CustomUTCOffset())
        date = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
        # Calculate the UTC offset using an instance of CustomUTCOffset
        utc_offset = CustomUTCOffset().utcoffset(date)
        # Combine the date and UTC offset
        formatted_date = date + utc_offset
        # Return only the date and time portion without microseconds or timezone
        return formatted_date.replace(microsecond=0, tzinfo=None)


## without CustomOffset, for general use:
# class DectimeConversion:
#     """
#     A class to convert decimal year to datetime object.
#     """

#     @staticmethod
#     def dectime_to_date(year, rem=None):
#         """
#         Convert a decimal year to a datetime object.
        
#         :param decimal_year: Decimal year (e.g., 2023.5 or 2023,5) or a tuple of two integers (e.g., (2023, 5))
#         :return: Corresponding datetime object
#         """
#         # Handle input as a single float or two integers
#         if rem is None:
#             # If only one argument is provided, treat it as a float
#             decimal_year = float(year)
#             year = int(decimal_year)
#             rem = decimal_year - year
#         else:
#             # If two arguments are provided, calculate rem from the second integer
#             rem = rem / 10**len(str(rem))

#         # Calculate the number of days in the year
#         if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
#             days_in_year = 366
#         else:
#             days_in_year = 365

#         # Calculate the number of seconds in the year and adjust for the decimal part
#         seconds_in_year = days_in_year * 24 * 60 * 60
#         base = datetime(year, 1, 1)
#         total_seconds = seconds_in_year * rem
#         date = base + timedelta(seconds=round(total_seconds))  # Round to the nearest second
#         return date
