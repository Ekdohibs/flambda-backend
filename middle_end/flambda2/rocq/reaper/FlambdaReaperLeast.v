(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* The least solution of the reaper's rules, computed, and a check that
   tables are the least solution.

   This file is the implementation, extracted for the test tool of the
   reaper (FlambdaExtract.v); it needs no review.
   FlambdaReaperLeastSpec.v says what its answers mean (the least
   solution there is [least_solution]: tables closed under the rules,
   every tuple of which the rules derive, the guards [~~(any_usage x)]
   and [~~(any_source x)] read off the tables themselves), and
   FlambdaReaperLeastProofs.v shows that they do.

   [step Tg T] applies every rule of FlambdaReaperGraphSpec.closed once
   to the tuples of [T], the guards read off [Tg], and adds what the
   rules conclude to [T], one pass over the facts and the tables for
   each rule, as FlambdaReaperSolution.v checks them. [saturate] steps
   until nothing is added: with [Tg] fixed the rules are monotone, and
   that is the least tables closed under them with those guards.
   [inflate] steps with the guards read off the tables so far, as the
   compiler's schedule reads them: a guard true when a rule fires may be
   false at the end, and the conclusion stays.

   [check_least T] tests that [T] is closed ([check_closed]) and that
   every tuple of [T] is derived from nothing with the guards read off
   [T] ([saturate T] has it).

   [compute_least] finds a least solution: the tables [inflate] ends
   with, then, until one is least, those that [saturate] gives with the
   guards of the last. Each is checked ([check_closed]) and the
   tuples it is made of are derived with guards that hold of it too
   ([tops_within]), so that what it returns is least, without a proof
   that it always returns. Every pass is quadratic or cubic in the
   facts and the tuples, which is enough for test units. *)

From Stdlib Require Import List Bool PeanoNat.
From OxCamlFlambda2 Require Import FlambdaReaperGraphSpec
  FlambdaReaperSolution.
Import ListNotations.

(* The tables with no tuple. *)
Definition empty_tables : tables := Tables [] [] [] [] [] [] [] [] [] [].

(* [l] if [b], and nothing otherwise. *)
Definition when {A : Type} (b : bool) (l : list A) : list A :=
  if b then l else [].

Section Step.

Variable is_local : field -> bool.
Variable F : facts.
(* The tables that the guards [~~(any_usage x)] and [~~(any_source x)]
   read. *)
Variable Tg : tables.
(* The tuples derived so far. *)
Variable T : tables.

Abbreviation alias to_ from := (mem_pair to_ from (t_alias T)).
Abbreviation any_usage x := (mem_node x (t_any_usage T)).
Abbreviation any_source x := (mem_node x (t_any_source T)).
Abbreviation has_usage x := (mem_node x (t_has_usage T)).
Abbreviation has_source x := (mem_node x (t_has_source T)).
Abbreviation top_usage x := (mem_node x (t_any_usage Tg)).
Abbreviation top_source x := (mem_node x (t_any_source Tg)).

(* What the rules of [alias_rules] conclude, in their order, the input
   first. *)
Definition new_alias : list (node * node) :=
  f_alias F
  ++ flat_map
       (fun '(if_used, to_, from) => when (any_usage if_used) [(to_, from)])
       (f_propagate F)
  ++ flat_map
       (fun '(if_any_source, to_, from) =>
          when (any_source if_any_source) [(to_, from)])
       (f_alias_if_any_source F)
  ++ flat_map
       (fun '(base, relation, from) =>
          when (negb (top_usage base))
            (flat_map
               (fun '(base', base_use) =>
                  when (node_eqb base base')
                    (flat_map
                       (fun '(to_, relation', base_use') =>
                          when (field_eqb relation relation'
                                && node_eqb base_use base_use')
                            [(to_, from)])
                       (f_accessor F)))
               (t_usages T)))
       (f_constructor F)
  ++ flat_map
       (fun '(base, relation, from) =>
          when (is_local relation)
            (flat_map
               (fun '(base', base_use) =>
                  when (node_eqb base base')
                    (flat_map
                       (fun '(to_, relation', base_use') =>
                          when (field_eqb relation relation'
                                && node_eqb base_use base_use')
                            [(to_, from)])
                       (f_accessor F)))
               (t_usages T)))
       (f_constructor F)
  ++ flat_map
       (fun '(base, relation, to_) =>
          when (negb (top_usage base))
            (flat_map
               (fun '(base', base_use) =>
                  when (node_eqb base base')
                    (flat_map
                       (fun '(from, relation', base_use') =>
                          when (Nat.eqb relation relation'
                                && node_eqb base_use base_use')
                            [(to_, from)])
                       (f_argument F)))
               (t_usages T)))
       (f_parameter F)
  ++ flat_map
       (fun '(to_, relation, base) =>
          when (negb (top_source base))
            (flat_map
               (fun '(base', base_source) =>
                  when (node_eqb base base')
                    (flat_map
                       (fun '(base_source', relation', from) =>
                          when (node_eqb base_source base_source'
                                && field_eqb relation relation')
                            [(to_, from)])
                       (f_constructor F)))
               (t_sources T)))
       (f_accessor F)
  ++ flat_map
       (fun '(to_, relation, base) =>
          when (is_local relation)
            (flat_map
               (fun '(base', base_source) =>
                  when (node_eqb base base')
                    (flat_map
                       (fun '(base_source', relation', from) =>
                          when (node_eqb base_source base_source'
                                && field_eqb relation relation')
                            [(to_, from)])
                       (f_constructor F)))
               (t_sources T)))
       (f_accessor F)
  ++ flat_map
       (fun '(relation, from) =>
          flat_map
            (fun '(relation', to_) =>
               when (field_eqb relation relation') [(to_, from)])
            (t_reading_field T))
       (t_escaping_field T)
  ++ flat_map
       (fun '(from, relation, base) =>
          when (negb (top_source base))
            (flat_map
               (fun '(base', base_source) =>
                  when (node_eqb base base')
                    (flat_map
                       (fun '(base_source', relation', to_) =>
                          when (node_eqb base_source base_source'
                                && Nat.eqb relation relation')
                            [(to_, from)])
                       (f_parameter F)))
               (t_sources T)))
       (f_argument F).

(* [has_usage_rules]. *)
Definition new_has_usage : list node :=
  t_any_usage T
  ++ flat_map (fun '(to_, from) => when (has_usage to_) [from]) (t_alias T)
  ++ flat_map (fun '(to_, _, base) => when (has_usage to_) [base])
       (f_accessor F)
  ++ flat_map (fun '(from, _, base) => when (has_source from) [base])
       (f_argument F).

(* [has_source_rules]. *)
Definition new_has_source : list node :=
  t_any_source T
  ++ flat_map (fun '(to_, from) => when (has_source from) [to_]) (t_alias T)
  ++ flat_map (fun '(base, _, from) => when (has_source from) [base])
       (f_constructor F)
  ++ flat_map
       (fun '(base, relation, _) => when (is_function_slot relation) [base])
       (f_constructor F)
  ++ flat_map (fun '(base, _, to_) => when (has_usage to_) [base])
       (f_parameter F).

(* [any_usage_rules], and the input [any_usage]. *)
Definition new_any_usage : list node :=
  flat_map (fun '(to_, from) => when (has_usage to_) [from]) (f_use F)
  ++ flat_map (fun '(to_, from) => when (any_usage to_) [from]) (t_alias T)
  ++ flat_map
       (fun '(base, relation, from) =>
          when (any_usage base && negb (is_local relation)) [from])
       (f_constructor F)
  ++ flat_map (fun '(from, _, base) => when (any_source base) [from])
       (f_argument F)
  ++ f_any_usage F.

(* [any_source_rules], and the input [any_source]. *)
Definition new_any_source : list node :=
  f_any_source F
  ++ f_zero_alloc_source F
  ++ flat_map (fun '(if_used, x) => when (any_usage if_used) [x])
       (f_any_source_if_used F)
  ++ flat_map (fun '(to_, from) => when (any_source from) [to_]) (t_alias T)
  ++ flat_map (fun '(base, _, to_) => when (any_usage base) [to_])
       (f_parameter F)
  ++ flat_map
       (fun '(to_, relation, base) =>
          when (any_source base && negb (is_local relation)) [to_])
       (f_accessor F)
  ++ flat_map (fun '(to_, from) => when (has_source from) [to_]) (f_use F).

(* [usages_rules], with the guard of [nontop_usages]. *)
Definition new_usages : list (node * node) :=
  flat_map (fun '(_, _, base) => when (negb (top_usage base)) [(base, base)])
    (f_accessor F)
  ++ flat_map
       (fun '(_, _, base) => when (negb (top_usage base)) [(base, base)])
       (f_argument F)
  ++ flat_map
       (fun '(to_, usage) =>
          when (negb (top_usage to_))
            (flat_map
               (fun '(to_', from) =>
                  when (node_eqb to_ to_' && negb (top_usage from))
                    [(from, usage)])
               (t_alias T)))
       (t_usages T).

(* [sources_rules], with the guard of [nontop_sources]. *)
Definition new_sources : list (node * node) :=
  flat_map
    (fun '(base, _, _) => when (negb (top_source base)) [(base, base)])
    (f_constructor F)
  ++ flat_map
       (fun '(base, _, _) => when (negb (top_source base)) [(base, base)])
       (f_parameter F)
  ++ flat_map
       (fun '(from, source) =>
          when (negb (top_source from))
            (flat_map
               (fun '(to_, from') =>
                  when (node_eqb from from' && negb (top_source to_))
                    [(to_, source)])
               (t_alias T)))
       (t_sources T).

(* [local_rules]. *)
Definition new_escaping_field : list (field * node) :=
  flat_map
    (fun '(base, relation, from) =>
       when (any_usage base && is_local relation) [(relation, from)])
    (f_constructor F).

Definition new_reading_field : list (field * node) :=
  flat_map
    (fun '(to_, relation, base) =>
       when (any_source base && is_local relation) [(relation, to_)])
    (f_accessor F).

(* [post_processing_rules]: [field_of_constructor_is_used]. *)
Definition new_field_used : list (node * field) :=
  flat_map
    (fun '(base, relation, _) =>
       when (any_usage base && negb (is_local relation)) [(base, relation)])
    (f_constructor F)
  ++ flat_map
       (fun '(base, relation, _) =>
          flat_map
            (fun '(base', usage) =>
               when (node_eqb base base')
                 (flat_map
                    (fun '(v, relation', usage') =>
                       when (field_eqb relation relation'
                             && node_eqb usage usage' && has_usage v)
                         [(base, relation)])
                    (f_accessor F)))
            (t_usages T))
       (f_constructor F)
  ++ flat_map
       (fun '(base, relation, _) =>
          when (any_usage base)
            (flat_map
               (fun '(relation', x) =>
                  when (field_eqb relation relation' && has_usage x)
                    [(base, relation)])
               (t_reading_field T)))
       (f_constructor F)
  ++ flat_map
       (fun '(base, relation, _) =>
          when (is_local relation && any_usage base)
            (flat_map
               (fun '(usage, base') =>
                  when (node_eqb base base')
                    (flat_map
                       (fun '(v, relation', usage') =>
                          when (field_eqb relation relation'
                                && node_eqb usage usage' && has_usage v)
                            [(base, relation)])
                       (f_accessor F)))
               (t_sources T)))
       (f_constructor F).

End Step.

(* [old] with the elements of [new] that it does not have. *)
Definition add_new {A : Type} (eqb : A -> A -> bool) (old new : list A)
    : list A :=
  fold_left (fun acc x => if existsb (eqb x) acc then acc else x :: acc)
    new old.

Definition pair_eqb (p p' : node * node) : bool :=
  node_eqb (fst p) (fst p') && node_eqb (snd p) (snd p').

Definition field_node_eqb (p p' : field * node) : bool :=
  field_eqb (fst p) (fst p') && node_eqb (snd p) (snd p').

Definition node_field_eqb (p p' : node * field) : bool :=
  node_eqb (fst p) (fst p') && field_eqb (snd p) (snd p').

(* The number of tuples of the tables. *)
Definition tables_size (T : tables) : nat :=
  length (t_alias T) + length (t_any_usage T) + length (t_any_source T)
  + length (t_has_usage T) + length (t_has_source T) + length (t_usages T)
  + length (t_sources T) + length (t_reading_field T)
  + length (t_escaping_field T) + length (t_field_used T).

(* Whether every tuple of [T] is in [T']. *)
Definition tables_incl (T T' : tables) : bool :=
  forallb (fun '(to_, from) => mem_pair to_ from (t_alias T')) (t_alias T)
  && forallb (fun x => mem_node x (t_any_usage T')) (t_any_usage T)
  && forallb (fun x => mem_node x (t_any_source T')) (t_any_source T)
  && forallb (fun x => mem_node x (t_has_usage T')) (t_has_usage T)
  && forallb (fun x => mem_node x (t_has_source T')) (t_has_source T)
  && forallb (fun '(x, y) => mem_pair x y (t_usages T')) (t_usages T)
  && forallb (fun '(x, y) => mem_pair x y (t_sources T')) (t_sources T)
  && forallb (fun '(f, x) => mem_field_node f x (t_reading_field T'))
       (t_reading_field T)
  && forallb (fun '(f, x) => mem_field_node f x (t_escaping_field T'))
       (t_escaping_field T)
  && forallb (fun '(x, f) => mem_node_field x f (t_field_used T'))
       (t_field_used T).

(* The guards that hold of [T'] hold of [T]: [T] has no [any_usage] nor
   [any_source] that [T'] has not. *)
Definition tops_within (T T' : tables) : bool :=
  forallb (fun x => mem_node x (t_any_usage T')) (t_any_usage T)
  && forallb (fun x => mem_node x (t_any_source T')) (t_any_source T).

Section Iterate.

Variable is_local : field -> bool.
Variable F : facts.

(* One application of every rule. *)
Definition step (Tg T : tables) : tables :=
  {| t_alias := add_new pair_eqb (t_alias T) (new_alias is_local F Tg T);
     t_any_usage :=
       add_new node_eqb (t_any_usage T) (new_any_usage is_local F T);
     t_any_source :=
       add_new node_eqb (t_any_source T) (new_any_source is_local F T);
     t_has_usage := add_new node_eqb (t_has_usage T) (new_has_usage F T);
     t_has_source :=
       add_new node_eqb (t_has_source T) (new_has_source F T);
     t_usages := add_new pair_eqb (t_usages T) (new_usages F Tg T);
     t_sources := add_new pair_eqb (t_sources T) (new_sources F Tg T);
     t_reading_field :=
       add_new field_node_eqb (t_reading_field T)
         (new_reading_field is_local F T);
     t_escaping_field :=
       add_new field_node_eqb (t_escaping_field T)
         (new_escaping_field is_local F T);
     t_field_used :=
       add_new node_field_eqb (t_field_used T)
         (new_field_used is_local F T) |}.

(* The rules with the guards read off [Tg], from [T], until nothing is
   added or [fuel] steps are made. *)
Fixpoint saturate (Tg : tables) (fuel : nat) (T : tables) : tables :=
  match fuel with
  | O => T
  | S fuel =>
    let T' := step Tg T in
    if Nat.eqb (tables_size T') (tables_size T) then T
    else saturate Tg fuel T'
  end.

(* The rules with the guards read off the tables so far, from [T]. *)
Fixpoint inflate (fuel : nat) (T : tables) : tables :=
  match fuel with
  | O => T
  | S fuel =>
    let T' := step T T in
    if Nat.eqb (tables_size T') (tables_size T) then T
    else inflate fuel T'
  end.

(* [T] is closed, and every tuple of it is derived by the rules, with
   the guards read off [T], in [fuel] steps. *)
Definition check_least (fuel : nat) (T : tables) : bool :=
  check_closed is_local F T
  && tables_incl T (saturate T fuel empty_tables).

(* The tables that the rules derive with the guards read off [Tg], if
   they are closed and hold no guard that [Tg] does not; otherwise, at
   most [rounds] times again, with the guards read off them. *)
Fixpoint alternate (fuel rounds : nat) (Tg : tables) : option tables :=
  let T := saturate Tg fuel empty_tables in
  if check_closed is_local F T && tops_within T Tg then Some T
  else
    match rounds with
    | O => None
    | S rounds => alternate fuel rounds T
    end.

(* A least solution of the rules, from the tables that the schedule of
   the compiler gives. [fuel] bounds the steps of each saturation and the
   rounds. *)
Definition compute_least (fuel : nat) : option tables :=
  alternate fuel fuel (inflate fuel empty_tables).

End Iterate.
