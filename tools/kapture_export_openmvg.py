#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to export a kapture into an openmvg file.
"""

import argparse
import logging
import sys
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture.utils.logging
from kapture.io.records import TransferAction
from kapture.converter.openmvg.export_openmvg import export_openmvg

logger = logging.getLogger('openmvg')


def export_openmvg_command_line():
    """
    Do the kapture to openmvg export using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='Exports from kapture format to openMVG JSON file.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output file if already exists.')
    # create the parser for the export command #########################################################################
    parser.add_argument('-i', '-k', '--kapture', required=True,
                        help='path to input kapture data root directory')
    parser.add_argument('-s', '--sfm_data', required=True,
                        help='path to output openMVG sfm_data JSON file.')
    parser.add_argument('-im', '--images',
                        help='path to output openMVG image directory.')
    parser.add_argument('-r', '--regions',
                        help='path to output openMVG regions directory (for features and descriptors).')
    parser.add_argument('-m', '--matches',
                        help='path to output openMVG matches file.')
    parser.add_argument('--image_action', default='root_link', type=TransferAction,
                        help=f'''what to do with images:
        {TransferAction.root_link.name}: link to the root of the images directory (default) ;
        {TransferAction.link_absolute.name}: absolute individual file link ;
        {TransferAction.link_relative.name}: relative individual file link ;
        {TransferAction.copy.name}: copy file instead of creating link ;
        {TransferAction.move.name}: move file instead of creating link ;
        {TransferAction.skip.name}: do not create links
                                      ''')

    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # for debug, let kapture express itself.
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    logger.debug(f'{sys.argv[0]} \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v))
        for k, v in vars(args).items()))

    # no image dir == sip transfer
    if args.image_action == TransferAction.skip:
        args.images = None
    if args.images is None:
        args.image_action = TransferAction.skip

    export_openmvg(
        kapture_path=args.kapture,
        openmvg_sfm_data_file_path=args.sfm_data,
        openmvg_image_root_path=args.images,
        openmvg_regions_dir_path=args.regions,
        openmvg_matches_file_path=args.matches,
        image_action=args.image_action,
        force=args.force
    )


if __name__ == '__main__':
    try:
        export_openmvg_command_line()
    except Exception:
        logger.error('Fatal error', exc_info=True)
