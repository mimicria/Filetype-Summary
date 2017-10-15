#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SYNOPSIS

	python files_in_there_but_not_here.py [-h,--help] [-v,--verbose]


DESCRIPTION

	Given a target directory and a list of other directories, list the files
    that exist within the other directories but are absent within the target.


ARGUMENTS

	-h, --help	        show this help message and exit
	-v, --verbose       verbose output


AUTHOR

	Doug McGeehan <djmvfb@mst.edu>


LICENSE

	Copyright 2017  - GNU GPLv3


TODO

    Implement interprocess-communicating threading, where the walking of the
    target directory communicates with the walkers of the other directories.

"""
import math
import humanize
from lib.lineheaderpadded import hr

__appname__ = 'files_in_there_but_not_here'
__version__ = '0.0pre0'
__license__ = 'GNU GPLv3'
__indev__ = True

import argparse
from datetime import datetime
import sys
import os
import collections
import logging
import itertools

logger = logging.getLogger(__name__)


def main(args):
    summary = DirectorySummary(root=args.target)
    summary.walk(valid_extensions=args.targeted_extensions)
    summary.print(to_file='extensions_summary.txt')

    if not args.skip_report:
        summary.plot()


def get_hex_colors(n):
    import colorsys
    HSV_tuples = [(x*1.0/n, 0.15, 0.75) for x in range(n)]
    colors = []
    for rgb in HSV_tuples:
        rgb = colorsys.hsv_to_rgb(*rgb)
        colors.append(tuple(rgb))
    return colors


class DirectorySummary(object):
    def __init__(self, root):
        # Assert existence of the directory and return its absolute path
        self.root = self._existing_directory(root)

        self.file_type_sizes = collections.defaultdict(list)
        self.directory_based_file_types = collections.defaultdict(dict)

        self.total_size = 0
        self.num_files = 0

    def _existing_directory(self, directory):
        """
        Assert directory exists. If it does, return the absolute path of the 
        directory without the appended forward slash "/"
        :return: Absolute path to the provided directory.
        """
        assert os.path.isdir(directory), '"{}" is not a directory.'.format(
                                             directory)

        # If the directory ends with a forward slash, remove it
        if directory[-1] == '/':
            directory = os.path.dirname(directory)

        return directory #os.path.abspath(directory)

    def walk(self, valid_extensions):
        logger.info('Walking through all files in "{}"'.format(self.root))
        for subdirectory, directory_names, files in os.walk(self.root):

            for filename in files:
                file_path = os.path.join(subdirectory, filename)

                # Skip over symbolic links.
                if os.path.islink(file_path):
                    continue

                filename_prefix, extension = os.path.splitext(filename)
                if valid_extensions is not None and \
                   not filename.lower().endswith(tuple(valid_extensions)):
                    continue

                if extension == '':
                    extension = '(none)'

                file_size = os.path.getsize(file_path)
                self.total_size += file_size

                self.file_type_sizes[extension].append(file_size)

                parent_directory = os.path.dirname(file_path)
                self.walk_path_to_root(extension, file_size, parent_directory)

                self.num_files += 1

    def walk_path_to_root(self, extension, file_size, parent_directory):
        ignore_abs_path = len(os.path.dirname(self.root))+1
        while parent_directory != self.root:
            directory_summary = self.directory_based_file_types[
                parent_directory[ignore_abs_path:]]

            if extension not in directory_summary:
                directory_summary[extension] = []

            directory_summary[extension].append(file_size)

            parent_directory = os.path.dirname(parent_directory)

    def print(self, to_file=None):
        cli_plot = CommandLineHorizontalPlot(data=self.file_type_sizes)
        cli_plot.plot(
            title='Space Allocation per Extension: {}'.format(self.root),
            max_value=self.total_size,
            aggregate_fn=sum,
            value_fmt_fn=lambda x: humanize.naturalsize(x, binary=True)
                                           .rjust(4),
            to_file=to_file
        )
        cli_plot.plot(
            title='File Counts per Extension: {}'.format(self.root),
            max_value= self.num_files,
            aggregate_fn=len,
            to_file=to_file
        )


    def plot(self):
        # partition directories by their dominating extension
        dominating_extensions = {k: [] for k in self.file_type_sizes.keys()}
        for directory_path, extension_stats in \
                self.directory_based_file_types.items():

            stats = DirectoryExtensionStats(path=directory_path,
                                            extension_stats=extension_stats)
            dominating_extensions[stats.dominating_ext].append(stats)

        logger.debug(hr('Dominating Extentions'))
        for extension in list(dominating_extensions.keys()):
            ext_stats = dominating_extensions[extension]
            if not ext_stats:
                del dominating_extensions[extension]
                continue

            ext_stats.sort(
                key=lambda stats: stats.proportion_files_with_dominating_ext,
                reverse=True)

            logger.debug(hr(extension, '-'))
            for stats in ext_stats:
                stats.sort_extensions()
                stats.summary()

        # Print out summary of the walked directories
        logger.info(hr('Summary'))

        num_unique_extensions = len(self.file_type_sizes)
        logger.info('Number of unique extension: {}'.format(
            num_unique_extensions))

        colors = get_hex_colors(n=num_unique_extensions)
        color_wheel = itertools.cycle(colors)
        num_subdirectories = len(self.directory_based_file_types)
        logger.info('{0} subdirectories contained within {1}'.format(
            num_subdirectories, self.root
        ))

        max_length_of_leaf_directory_path = len(
            max(self.directory_based_file_types.keys(),
                key=len))
        logger.debug('Max length of directory path: {} chars'.format(
            max_length_of_leaf_directory_path))

        i = 0
        max_stats_per_page = 125
        report_filename_format = 'extension_breakdown_p{:0>6}.pdf'
        page_num = 0
        extension_stats = []
        report_pages = []

        logger.info('{} pages will be created'.format(
            int(math.ceil(num_subdirectories/max_stats_per_page)+1)))

        report_directory = 'report_pages'
        file_extension_report_path = os.path.join(report_directory,
                                                  'filetype_breakdown.pdf')
        logger.info('File type report will be stored in "{}"'.format(
            file_extension_report_path))
        os.makedirs(report_directory, exist_ok=True)

        for ext, ext_stats_by_dominating_stats in dominating_extensions.items():

            for stats in ext_stats_by_dominating_stats:
                if i == max_stats_per_page:
                    logger.debug('Number of bars: {}'.format(
                        len(extension_stats)))
                    plot = DirectoryBreakdownFigure(
                        extension_stats=extension_stats,
                        margin_width=max_length_of_leaf_directory_path,
                        plot_height=max_stats_per_page,
                        color_map=color_wheel
                    )

                    report_filename = os.path.join(
                        report_directory,
                        report_filename_format.format(page_num))
                    plot.plot(save_to=report_filename,
                              walked_directory=self.root)

                    report_pages.append(report_filename)
                    i = 0
                    page_num += 1
                    extension_stats = []

                extension_stats.append(stats)
                i += 1

        if i <= max_stats_per_page and i > 0:
            logger.info('Exporting final pdf file')
            plot = DirectoryBreakdownFigure(
                extension_stats=extension_stats,
                margin_width=max_length_of_leaf_directory_path,
                plot_height=max_stats_per_page,
                color_map=color_wheel
            )

            if page_num == 0:
                page_num = 1

            report_filename = report_filename_format.format(page_num)
            plot.plot(save_to=report_filename,
                      walked_directory=self.root)
            report_pages.append(report_filename)

        from PyPDF2 import PdfFileMerger, PdfFileReader

        merger = PdfFileMerger()
        for filename in report_pages:
            merger.append(PdfFileReader(open(filename, 'rb')))


        merger.write(file_extension_report_path)

        for filename in report_pages:
            os.remove(filename)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
class DirectoryBreakdownFigure(object):
    bar_colors = {}
    #background_color_wheel = itertools.cycle(['#00000000', '#11111100'])

    def __init__(self, extension_stats, margin_width, plot_height, color_map):
        self.extension_stats = extension_stats
        self.margin_width = margin_width
        self.plot_height = plot_height
        self.color_map = color_map

    def plot(self, save_to, walked_directory):
        logger.debug('Creating page: {}'.format(save_to))
        verticle_space = int(self.plot_height/6)
        horizontal_space = int(self.margin_width/10) + 3

        logger.debug('Verticle space:   {}'.format(verticle_space))
        logger.debug('Horizontal space: {}'.format(horizontal_space))

        figure, axes = plt.subplots(
            figsize=(horizontal_space, verticle_space))

        directory_labels, right_y_axis = self.construct_plot(axes)

        self.format_plot(figure, axes, directory_labels, right_y_axis,
                         walked_directory)
        plt.savefig(save_to)
        plt.close()

    def format_plot(self, figure, axes, directory_labels, right_y_axis,
                    directory):
        # add directories to the right of the plot
        y_ticks = numpy.arange(len(directory_labels)) + 0.5
        right_y_axis.set_yticks(y_ticks)
        right_y_axis.set_yticklabels(
            directory_labels,)
            # backgroundcolor=[next(self.background_color_wheel)
            #                  for _ in range(len(directory_labels))])
        right_y_axis.tick_params(axis='y', which='both', length=0)
        axes.set_xlim([0, 1])
        axes.set_ylim([0, self.plot_height])
        right_y_axis.set_ylim([0, self.plot_height])
        #axes.invert_xaxis()
        right_y_axis.invert_yaxis()
        # hide the tickmarks on the left y-axis
        axes.set_yticks([])
        axes.set_xticks([])

        # make the border of the plots invisible
        for side in ['top', 'right', 'bottom', 'left']:
            axes.spines[side].set_visible(False)
            right_y_axis.spines[side].set_visible(False)

        figure.suptitle('File Type Breakdown: {}'.format(directory))
        plt.tight_layout(pad=9.25)

    def construct_plot(self, axes):
        # create the bar for each directory
        directory_labels = []
        y_val = 0
        right_y_axis = axes.twinx()

        for directory_stats in self.extension_stats:
            directory_labels.append(directory_stats.path)

            bar_widths, bar_offsets, colors, annotations = self.single_barh(
                ext_stats=directory_stats)

            # every horizontal bar created for this directory will be
            #  located on the same y height
            num_bars = len(bar_widths)
            y_vals = numpy.zeros(num_bars) + y_val + .5
            right_y_axis.barh(bottom=y_vals,
                              width=bar_widths,
                              height=1,
                              left=bar_offsets,
                              color=colors,
                              # linewidth=0)
                              edgecolor='black')

            # for text_x, ext, color in annotations:
            for text_x, ext in annotations:
                right_y_axis.text(
                    x=text_x,
                    y=y_val + .20,
                    s=ext,
                    fontsize=8,
                    horizontalalignment='left',
                    verticalalignment='top')

            y_val += 1

        logger.debug('{} bars produced'.format(y_val))
        logger.debug('{} directories considered'.format(self.plot_height))
        return directory_labels, right_y_axis

    def single_barh(self, ext_stats):
        bar_widths = []
        bar_offsets_from_left = [0]
        colors = []
        ext_annotations = []
        for ext, proportion in ext_stats:
            if ext not in DirectoryBreakdownFigure.bar_colors:
                DirectoryBreakdownFigure.bar_colors[ext] = next(self.color_map)

            ext_color = DirectoryBreakdownFigure.bar_colors[ext]
            colors.append(ext_color)
            bar_widths.append(proportion)

            # set the offet for the bar to be drawn after this one
            start_of_bar = bar_offsets_from_left[-1]
            end_of_bar = proportion + start_of_bar
            bar_offsets_from_left.append(end_of_bar)

            extention_width = len(ext)*0.023
            if extention_width < proportion:
                text_x = start_of_bar + 0.01
                ext_annotations.append((text_x, ext))

        # remove the last offset, as it doesn't correspond to any extension
        # due to the manner in which the bar offsets are created
        bar_offsets_from_left.pop()

        return bar_widths, bar_offsets_from_left, colors, ext_annotations



class DirectoryExtensionStats(object):
    def __init__(self, path, extension_stats):
        self.path = path
        self.extension_stats = extension_stats

        ext, count, space, proportional_count = self._determine_dominating_ext(
            extension_stats)
        self.dominating_ext = ext
        self.num_files_with_dominating_ext = count
        self.space_allocated_to_dominating_ext = space
        self.proportion_files_with_dominating_ext = proportional_count

    def _determine_dominating_ext(self, extension_stats):
        # determine the extension that dominates this path
        ext = max(extension_stats.keys(),
                  key=lambda k: len(extension_stats[k]))
        count = len(extension_stats[ext])
        space = sum(extension_stats[ext])

        # count the total number of files within this directory
        ext_count_pairs = map(lambda k: (k, len(extension_stats[k])),
                              extension_stats.keys())
        self.num_files_within_dir = sum(map(lambda v: v[1],
                                       ext_count_pairs))
        # record the proportion of files within this directory that have the
        # dominating extension
        proportion = count / self.num_files_within_dir
        return ext, count, space, proportion

    def summary(self):
        # logger.debug('Dominated by: {0: >7} - {1: >6}, {2: >4}'.format(
        #         self.dominating_ext,
        #         '{:.1%}'.format(self.proportion_files_with_dominating_ext),
        #         filesize.size(self.space_allocated_to_dominating_ext,
        #                       system=filesize.si)
        #     ))

        logger.debug('┌─────────┬─────────┬────────┐  ' + self.path)
        for ext, portion in self.sorted_extensions.items():
            logger.debug('│ {0: ^7} │ {1: ^7} │  {2: >4}  │'.format(
                ext,
                '{:.1%}'.format(portion),
                humanize.naturalsize(sum(self.extension_stats[ext]),
                                     binary=True)
            ))
        logger.debug('└─────────┴─────────┴────────┘')

    def sort_extensions(self):
        ext_portion_pair = map(lambda x: (x, len(self.extension_stats[x])),
                               self.extension_stats.keys())
        ext_portion_pair = sorted(ext_portion_pair,
                                  key=lambda x: (x[1], x[0]),
                                  reverse=True)
        sorted_extensions = collections.OrderedDict()
        for ext, portion in ext_portion_pair:
            sorted_extensions[ext] = portion/self.num_files_within_dir

            self.sorted_extensions = sorted_extensions

    def __iter__(self):
        for extension, proportion in self.sorted_extensions.items():
            yield (extension, proportion)

class CommandLineHorizontalPlot(object):
    def __init__(self, data):
        self.data = data
        self.max_key_length = len(max(data.keys(), key=len))

    def plot(self, max_value, title, aggregate_fn, value_fmt_fn=None,
             to_file=None):
        if value_fmt_fn is None:
            value_fmt_fn = lambda x: x

        title = self.generate_title(title)
        top_border = self.generate_horizontal_border(corner='┌')
        plot_lines = self.generate_internal_plotlines(data=self.data,
                                                      max_value=max_value,
                                                      aggregate_fn=aggregate_fn,
                                                      value_fmt_fn=value_fmt_fn)
        bottom_border = self.generate_horizontal_border(corner='└')

        plot_content = [title, top_border]
        plot_content.extend(plot_lines)
        plot_content.append(bottom_border)
        """
        plot_content = '\n'.join([
            title,
            top_border,
            *plot_lines,
            bottom_border,
        ]) + '\n'
        """
        plot_content = '\n'.join(plot_content)
        print(plot_content)

        if to_file:
            with open(to_file, 'a') as f:
                f.write(plot_content)
                f.write('\n')


    def generate_title(self, title):
        title = '{margin}   {title}'.format(
            margin=self.margin(),
            title=title,
        )
        return title

    def margin(self, key=None):
        if key is None:
            key = ' '

        return key.rjust(self.max_key_length)

    def generate_horizontal_border(self, corner):
        border = '{margin} {corner}{border}┤'.format(
            margin=self.margin(),
            corner=corner,
            border='─' * 100,
        )
        return border

    def generate_internal_plotlines(self, data, max_value, aggregate_fn,
                                    value_fmt_fn):
        lines = []

        sorted_and_aggregated_data = sorted(
            [(k, aggregate_fn(data[k])) for k in data],
            key=lambda x: x[1],
            reverse=True
        )
        for key, value in sorted_and_aggregated_data:
            percentage = value / max_value
            percentage_string = '{:.1%}'.format(percentage)\
                                        .rjust(5)
            formatted_value = value_fmt_fn(value)

            lines.append('{margin} │{bar}│'
                         ' {percentage}, {value}'.format(
                margin=self.margin(key=key),
                bar=self.internal_data_line(percentage=percentage),
                percentage=percentage_string,
                value=formatted_value,
            ))

        return lines

    def internal_data_line(self, percentage, width=100, char='+'):
        data_chars = '+' * int(width*percentage)
        padded_data_line = data_chars.ljust(width)
        return padded_data_line



def setup_logger(args):
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    # todo: place them in a log directory, or add the time to the log's
    # filename, or append to pre-existing log
    log_file = os.path.join('/tmp', '.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    if args.verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    line_numbers_and_function_name = logging.Formatter(
        "%(levelname)s [%(filename)s:%(lineno)s - %(funcName)20s() ] "
        "%(message)s")
    fh.setFormatter(line_numbers_and_function_name)
    # ch.setFormatter(line_numbers_and_function_name)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Description printed to command-line if -h is called."
    )
    # during development, I set default to False so I don't have to keep
    # calling this with -v
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False,
                        help='Enable debugging messages (default: False)')
    parser.add_argument('--skip-report', action='store_true',
                        default=False,
                        help='Skip creation of PDF report')
    parser.add_argument('target', metavar='TARGET_DIR',
                        help='The target directory to search')
    parser.add_argument('-x', '--extensions', dest='targeted_extensions',
                        metavar='.EXT', nargs='*',
                        help='Extensions to focus on, ignoring all others. '
                             'Make sure they are lower case!')

    args = parser.parse_args()
    return args


def log_args(args):
    # figure out which argument key is the longest so that all the
    # parameters can be printed out nicely
    logger.debug('Command-line arguments:')
    length_of_longest_key = len(max(vars(args).keys(),
                                    key=lambda k: len(k)))
    for arg in vars(args):
        value = getattr(args, arg)
        logger.debug('\t{argument_key}:\t{value}'.format(
            argument_key=arg.rjust(length_of_longest_key, ' '),
            value=value))


if __name__ == '__main__':
    try:
        start_time = datetime.now()

        args = get_arguments()
        setup_logger(args)
        log_args(args)

        logger.debug(start_time)

        main(args)

        finish_time = datetime.now()
        logger.debug(finish_time)
        logger.debug('Execution time: {time}'.format(
            time=(finish_time - start_time)
        ))
        logger.debug("#" * 20 + " END EXECUTION " + "#" * 20)

        sys.exit(0)

    except KeyboardInterrupt as e:  # Ctrl-C
        raise e

    except SystemExit as e:  # sys.exit()
        raise e

    except Exception as e:
        logger.exception("Something happened and I don't know what to do D:")
        sys.exit(1)
