#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    30/01/2017
#
# *************************************** #


import time
import logging
import configparser
import re
from uuid import uuid4

from telegram import InlineQueryResultArticle, ParseMode, \
    InputTextMessageContent, ChatAction, Chat, \
    InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

from models.src.preprocessing import NLPPreprocessing


logging.getLogger(__name__).addHandler(logging.NullHandler())


class NLPUtil:
    BACK = '<< Back'
    UNKNOWN = 'UNK'


class NLPLng:
    RUS = 'RUS'
    ENG = 'ENG'
    UNK = NLPUtil.UNKNOWN

    @staticmethod
    def get_str(lng):
        if lng == NLPLng.RUS:
            return 'Русский'

        if lng == NLPLng.ENG:
            return 'English'

        return 'Unknown'


class NLPMode:
    SENTI = 'sentiment'
    CLASS = 'classification'
    KEYWS = 'search'
    UNKNW = NLPUtil.UNKNOWN

    @staticmethod
    def get_str(mode):
        if mode == NLPMode.SENTI:
            return 'Sentiment Analysis'

        if mode == NLPMode.CLASS:
            return 'Classification'

        if mode == NLPMode.KEYWS:
            return 'Key Word Search'

        return 'Unknown'


class NLPBot(object):
    def __init__(self, config_file=None):
        if config_file is None:
            self.config_file = 'bot.ini'

        config = configparser.ConfigParser()
        config.read(self.config_file)

        self.token = config['KEYS']['bot_api']
        self.admin_id = config['ADMIN']['id']
        self.logger = logging.getLogger(__name__)

        self.language = NLPLng.RUS
        self.mode = NLPMode.SENTI

        self.nlp_pre = NLPPreprocessing()

    @property
    def get_token(self):
        return self.token

    @property
    def get_admin_id(self):
        return self.admin_id

    def start_handler(self, bot, update):
        # reply_keyboard = [[NLPMode.get_str(NLPMode.SENTI),
        #                    NLPMode.get_str(NLPMode.CLASS),
        #                    NLPMode.get_str(NLPMode.KEYWS)]]

        update.message.reply_text('<b>I\'m NLP bot.</b>\n'
                                  'I can analyze your text messages.\n\n'
                                  'Default language is %s.\n'
                                  'Default mode is %s.\n\n'
                                  '<b>Available modes:</b>\n'
                                  '    - Sentiment Analysis\n'
                                  '    - Classification\n'
                                  '    - Key Word Search.\n\n'
                                  '<b>Commands:</b>\n'
                                  '/mode - select type of analysis\n'
                                  '/help - show help message\n'
                                  '\n'
                                  'Let\'s start!' %
                                  (NLPLng.get_str(self.language),
                                   NLPMode.get_str(self.mode)),
                                  parse_mode=ParseMode.HTML
                                  )

    def help_handler(self, bot, update):
        update.message.reply_text('<b>Help!</b> I need somebody ...\n'
                                  'Set a mode (<b>Current:</b> %s).\n'
                                  'Then send me a message.' %
                                  NLPMode.get_str(self.mode),
                                  parse_mode=ParseMode.HTML)

    def lng_handler(self, bot, update):
        keyboard = [[InlineKeyboardButton(NLPLng.get_str(NLPLng.RUS), callback_data=NLPLng.RUS),
                     InlineKeyboardButton(NLPLng.get_str(NLPLng.ENG), callback_data=NLPLng.ENG)]]

        lng_markup = InlineKeyboardMarkup(keyboard)

        update.message.reply_text('Set language. <b>Current:</b> %s' % NLPLng.get_str(self.language),
                                  reply_markup=lng_markup,
                                  parse_mode=ParseMode.HTML)


    def mode_handler(self, bot, update):
        keyboard = [[InlineKeyboardButton(NLPMode.get_str(NLPMode.SENTI), callback_data=NLPMode.SENTI),
                     InlineKeyboardButton(NLPMode.get_str(NLPMode.CLASS), callback_data=NLPMode.CLASS),
                     InlineKeyboardButton(NLPMode.get_str(NLPMode.KEYWS), callback_data=NLPMode.KEYWS)]]

        mode_markup = InlineKeyboardMarkup(keyboard)

        update.message.reply_text('Set mode. <b>Current:</b> %s.' % NLPMode.get_str(self.mode),
                                  reply_markup=mode_markup,
                                  parse_mode=ParseMode.HTML)

    def escape_markdown(self, text):
        """Helper function to escape telegram markup symbols"""
        escape_chars = '\*_`\['
        return re.sub(r'([%s])' % escape_chars, r'\\\1', text)

    def inline_handler(self, bot, update):
        query = update.inline_query.query
        results = list()

        results.append(InlineQueryResultArticle(id=uuid4(),
                                                title=NLPMode.get_str(NLPMode.SENTI),
                                                input_message_content=InputTextMessageContent(
                                                    query.upper())))

        results.append(InlineQueryResultArticle(id=uuid4(),
                                                title=NLPMode.get_str(NLPMode.CLASS),
                                                input_message_content=InputTextMessageContent(
                                                    "*%s*" % self.escape_markdown(query),
                                                    parse_mode=ParseMode.MARKDOWN)))

        results.append(InlineQueryResultArticle(id=uuid4(),
                                                title=NLPMode.get_str(NLPMode.KEYWS),
                                                input_message_content=InputTextMessageContent(
                                                    "_%s_" % self.escape_markdown(query),
                                                    parse_mode=ParseMode.MARKDOWN)))

        update.inline_query.answer(results)

    def button_handler(self, bot, update):
        query = update.callback_query

        self.logger.debug('callback_query: %s', query)

        if query.data in (NLPLng.ENG, NLPLng.RUS):
            reply_text = '<b>Language:</b> %s' % NLPLng.get_str(query.data)
            self.language = query.data
            self.logger.info('Chat: %s Language: %s',
                             query.message.chat_id,
                             query.data)
        elif query.data in (NLPMode.KEYWS, NLPMode.CLASS, NLPMode.SENTI):
            reply_text = '<b>Mode:</b> %s' % NLPMode.get_str(query.data)
            self.mode = query.data
            self.logger.info('Chat: %s Mode: %s',
                             query.message.chat_id,
                             query.data)
        else:
            reply_text = 'Unknown answer ...'
            self.logger.info('Chat: %s Answer: %s',
                             query.message.chat_id,
                             query.data)

        bot.editMessageText(text=reply_text,
                            chat_id=query.message.chat_id,
                            message_id=query.message.message_id,
                            parse_mode=ParseMode.HTML)

        # bot.answerCallbackQuery(callback_query_id=query.id,
        #                         text=reply_text)

    def text_handler(self, bot, update):
        try:
            if update.message.from_user.id != int(self.admin_id):
                bot.sendChatAction(chat_id=update.message.chat_id,
                                   action=ChatAction.TYPING)
                time.sleep(1)
                bot.sendMessage(chat_id=update.message.chat_id,
                                text="You aren't allowed to use bot yet. Try later or contact the author.")
                return

            self.logger.info('Chat: %s Language: %s Mode: %s',
                             update.message.chat_id,
                             self.language,
                             self.mode)

            # Get the text from user
            text = update.message.text

            # Detect language
            text_lng = self.nlp_pre.detect_language(text)

            self.logger.info('Chat: %s Language: %s Mode: %s Text language: %s',
                             update.message.chat_id,
                             self.language,
                             self.mode,
                             text_lng)

            bot.sendChatAction(chat_id=update.message.chat_id,
                               action=ChatAction.TYPING)

            # Run analysis
            time.sleep(2)

            # Send reply
            chat_obj = bot.getChat(chat_id=update.message.chat_id)

            if text_lng == 'ru':
                reply_text = '<b>Пользователь:</b> ' + chat_obj.first_name.encode('utf-8') + '\n'
                reply_text += '<b>Язык:</b> Русский'
            elif text_lng == 'en':
                reply_text = '<b>User:</b> ' + chat_obj.first_name + '\n'
                reply_text += '<b>Language:</b> English'
            else:
                reply_text = '<b>User:</b> ' + chat_obj.first_name + '\n'
                reply_text += '<b>Language:</b> Unknown'

            update.message.reply_text(reply_text,
                                      quote=True,
                                      parse_mode=ParseMode.HTML)

        except UnicodeEncodeError:
            bot.sendMessage(chat_id=update.message.chat_id,
                            text="Sorry, but I can't answer.")
