FROM php:8.2-fpm

RUN apt update \
    && apt install -y zlib1g-dev g++ git libicu-dev zip libzip-dev zip \
    && docker-php-ext-install intl opcache pdo pdo_mysql \
    && pecl install apcu \
    && docker-php-ext-enable apcu \
    && docker-php-ext-configure zip \
    && docker-php-ext-install zip

WORKDIR /var/www/ideasegment/

RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

RUN curl -sS https://get.symfony.com/cli/installer | bash
RUN mv /root/.symfony5/bin/symfony /usr/local/bin/
#CMD symfony new --demo project --no-git && chown -R :www-data project/ && chmod -R 775 project/ && php-fpm
RUN git config --global user.email "alper.anapali@ideasoft.com.tr" \ 
    && git config --global user.name "alper.anapali"

#RUN composer install volume mount edildigi icin duzgun calismiyor containera girip yaz
