import paddle
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import datetime
import logging
import time
from collections import defaultdict, deque
from minigpt4.common import dist_utils


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not paddle.distributed.is_initialized():
            return
        t = paddle.to_tensor([self.count, self.total], dtype=paddle.float32)
        all_list = []
        paddle.distributed.all_gather(all_list, t)
        self.count = 0
        self.total = 0.0
        for i in all_list:
            self.count += i[0]
            self.total += i[1]
        self.count = self.count.numpy()[0]
        self.total = self.total.numpy()[0]
        
    @property
    def median(self):
        d = paddle.to_tensor(data=list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = paddle.to_tensor(data=list(self.deque), dtype='float32')
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg
            =self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append('{}: {}'.format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append('{}: {:.4f}'.format(name, meter.global_avg))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}',
            '{meters}', 'time: {time}', 'data: {data}']
        if paddle.device.cuda.device_count() >= 1:
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if paddle.device.cuda.device_count() >= 1:
                    print(log_msg.format(i, len(iterable), eta=eta_string,
                        meters=str(self), time=str(iter_time), data=str(
                        data_time), memory=paddle.device.cuda.
                        max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string,
                        meters=str(self), time=str(iter_time), data=str(
                        data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
            total_time_str, total_time / len(iterable)))


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def setup_logger():
    logging.basicConfig(level=logging.INFO if dist_utils.is_main_process() else
        logging.WARN, format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()])
