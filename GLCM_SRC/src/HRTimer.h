#ifndef HRTIMER_H
#define HRTIMER_H

#include <windows.h>

typedef struct {
	LARGE_INTEGER start;
	LARGE_INTEGER stop;
} stopWatch;

void startTimer(stopWatch *timer);

void stopTimer(stopWatch *timer);

double getElapsedTimeMillis(stopWatch *timer);

double getElapsedTimeSecs(stopWatch *timer);

#endif