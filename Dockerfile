FROM debian:unstable as init-opam

# --- # SETUP PACKAGES

RUN apt update -yqq
RUN apt upgrade -yqq
RUN DEBIAN_FRONTEND=noninteractive apt install git apt-utils pkg-config make m4 gcc patch unzip bubblewrap autoconf -yqq
RUN apt install opam -yqq

# --- # INSTALL AND INIT OPAM

RUN groupadd -r user && useradd --no-log-init -r -m -g user user
RUN usermod -a -G root,user user
USER user
RUN opam init --bare --disable-sandboxing -y
RUN opam switch create flambda-bootstrap ocaml-base-compiler.4.14.1
RUN opam update
RUN eval $(opam env --switch=flambda-bootstrap)
RUN opam install menhir.20210419 dune.3.8.1 -y

# --- # installing project

WORKDIR /home/user
COPY --chown=user:user . .
RUN eval $(opam env --switch=flambda-bootstrap)
