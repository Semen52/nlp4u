#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    30/01/2017
#
# *************************************** #


"""
This Bot uses the Updater class to handle the bot.
A few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Inline bot analyzes texts.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""


import logging

from telegram.ext import (Updater, InlineQueryHandler, CallbackQueryHandler, CommandHandler,
                          MessageHandler, Filters)
from telegram.error import (TelegramError, Unauthorized, BadRequest,
                            TimedOut, ChatMigrated, NetworkError)
# from telegram import MessageEntity

from nlp import NLPBot


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def error_callback(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))
    try:
        raise error
    except Unauthorized:
        # remove update.message.chat_id from conversation list
        pass
    except BadRequest:
        # handle malformed requests - read more below!
        pass
    except TimedOut:
        # handle slow connection problems
        pass
    except NetworkError:
        # handle other connection problems
        pass
    except ChatMigrated as e:
        # the chat_id of a group has changed, use e.new_chat_id instead
        pass
    except TelegramError:
        # handle all other telegram related errors
        pass


def main():
    nlp_bot = NLPBot()

    logger.info('API: %s', nlp_bot.get_token)

    updater = Updater(token=nlp_bot.get_token)
    dispatcher = updater.dispatcher

    # Command handlers
    dispatcher.add_handler(CommandHandler("start", nlp_bot.start_handler))
    dispatcher.add_handler(CommandHandler("help", nlp_bot.help_handler))
    # dispatcher.add_handler(CommandHandler("lng", nlp_bot.lng_handler))
    dispatcher.add_handler(CommandHandler("mode", nlp_bot.mode_handler))

    # Inline buttons handler
    dispatcher.add_handler(CallbackQueryHandler(nlp_bot.button_handler))

    # Text message handler
    dispatcher.add_handler(MessageHandler(Filters.text,
                                          nlp_bot.text_handler))
    # & (Filters.entity(MessageEntity.TEXT_MENTION) | Filters.entity(MessageEntity.TEXT_LINK))


    # Inline message handler
    dispatcher.add_handler(InlineQueryHandler(nlp_bot.inline_handler))

    # Log all errors
    dispatcher.add_error_handler(error_callback)

    # Start the Bot
    updater.start_polling()

    # Block until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
