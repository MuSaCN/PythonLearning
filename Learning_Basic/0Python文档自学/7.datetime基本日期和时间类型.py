# Author:Zhang Yuan
#timedelta对象表示时间的间隔，即两个日期或时间之间的差值。
from datetime import timedelta
d=timedelta()
print(d.days, d.seconds, d.microseconds)
print(timedelta.min)
print(timedelta.max)
timedelta.resolution
timedelta.days
timedelta.seconds
timedelta.microseconds

year=timedelta(days=365)
another_year=timedelta(weeks=40,days=84,hours=23,minutes=50,seconds=600) # adds up to 365 days
year.total_seconds()
another_year.total_seconds()
year == another_year
ten_years = 10 * year
ten_years, ten_years.days // 365
nine_years = ten_years - year
nine_years, nine_years.days // 365
three_years = nine_years // 3;
three_years, three_years.days // 365
abs(three_years - ten_years) == 2 * three_years + year

#date对象表示理想化日历中的日期（年、月和日），即当前公历日历在两个方向上无限延伸。
from datetime import date
date(2019,12,12)
date.today()
date.fromtimestamp(10000)
date.fromordinal(735678)
date.min
date.max
date.resolution
date.year
date.month
date.day
date1=date.today()
date1
delta=timedelta(days=10)
date2=date1+delta
date2
delta2=date2-date1
delta2
date2.replace(day=26)
date1.timetuple()
date1.toordinal()
date1.weekday()
date1.isoweekday()
date1.isocalendar()
date1.isoformat()
date1.__str__()
date1.ctime()
date1.strftime("%a")
date1.__format__("%a")

import time
from datetime import date
today=date.today()
today==date.fromtimestamp(time.time())
my_birthday=date(today.year,6,24)

if my_birthday < today:
    my_birthday=my_birthday.replace(year=today.year+1)
my_birthday
time_to_birthday=abs(my_birthday-today)
time_to_birthday.days

from datetime import date
d=date.fromordinal(730920)
d
t=d.timetuple()
for i in t:
    print(i)
ic=d.isocalendar()
for i in ic:
    print(i)
d.isoformat()
d.strftime("%d/%m/%y")
d.strftime("%A %d. %B %Y")
'The {1} is {0:%d}, the {2} is {0:%B}.'.format(d, "day", "month")

#datetime对象是一个包含date对象和time对象所有信息的单个对象。
from datetime import datetime
t1=datetime(2019,5,18)
datetime.today()
datetime.now()
datetime.utcnow()
datetime.fromtimestamp(time.time())
datetime.utcfromtimestamp(time.time())
datetime.fromordinal(10000)
from datetime import time
datetime.combine(date(2019,1,5),time(12,30))
datetime.strptime("11/03/19","%d/%m/%y")
datetime.min
datetime.max
datetime.resolution
t1.year
t1.month
t1.day
t1.hour
t1.microsecond
t1.tzinfo
t1.date()
t1.time()
t1.timetz()
t1.replace(2010)
t1.date()
t1.astimezone()
print(t1.utcoffset())
t1.timetuple()
t1.utctimetuple()
t1.toordinal()
t1.timestamp()
t1.weekday()
t1.isoweekday()
t1.ctime()
from datetime import datetime, date, time
d=date(2005,7,14)
t=time(12,30)
datetime.combine(d,t)
datetime.now()
datetime.utcnow()
dt = datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M")
dt
tt=dt.timetuple()
for it in tt:
    print(it)
ic=dt.isocalendar()
for it in ic:
    print(it)
dt.strftime("%A, %d. %B %Y %I:%M%p")
'The {1} is {0:%d}, the {2} is {0:%B}, the {3} is {0:%I:%M%p}.'.format(dt, "day", "month", "time")


#time对象表示一天中的（本地）时间，独立于任何特定的日子，并且可以通过tzinfo对象进行调整。
from datetime import time
t1=time(22,51,14,1000)
time.min
time.max
time.resolution
t1.hour
t1.isoformat()
t1.__str__()
t1.utcoffset()
t1.dst()
from datetime import time, tzinfo
class GMT1(tzinfo):
    def utcoffset(self, dt):
        return timedelta(hours=1)
    def dst(self, dt):
        return timedelta(0)
    def tzname(self,dt):
        return "Europe/Prague"
t = time(12, 10, 30, tzinfo=GMT1())
t




