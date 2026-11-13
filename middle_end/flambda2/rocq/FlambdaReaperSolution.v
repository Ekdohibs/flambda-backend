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

(* The check that a solution of the reaper's Datalog rules is closed under
   them.

   This file is the implementation, extracted for the test tool of the
   reaper (FlambdaExtract.v); it needs no review.
   FlambdaReaperSolutionSpec.v says what its answers mean, and
   FlambdaReaperSolutionProofs.v shows that they do.

   The solution is read off the compiler's Datalog database as tables
   (FlambdaReaperGraphSpec.tables), and nothing is believed of it: this
   file tests it. [check_closed] tests every rule of
   FlambdaReaperGraphSpec.closed, which are those of
   middle_end/flambda2/reaper/points_to_analysis.ml, with one test per
   rule, named after the field of [closed] that states it, and the tests
   of a group of rules of points_to_analysis.ml ([alias_rules],
   [has_usage_rules], ...) put together as there. A rule speaks of every
   node; the test goes over the facts of the graph and the tuples of the
   tables that its premises name, joined on the nodes and fields that
   they share, and asks of the tables, at each match, the conclusion of
   the rule. The negated premises ([~~(any_usage x)] in
   [nontop_usages]) are tests of the tables too. Each test is a pass
   over a list, or two or three nested: quadratic or cubic, which is
   enough for test units.

   [check_field_used_exact] tests that every tuple of
   [field_of_constructor_is_used] is given by one of the rules of
   [post_processing_rules] (FlambdaReaperGraphSpec.field_used_exact),
   one test per rule. *)

From Stdlib Require Import List Bool PeanoNat.
From OxCamlFlambda2 Require Import FlambdaReaperGraphSpec.
Import ListNotations.

Section Checks.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.

(* The relations of the solution, as the tables have them. *)
Abbreviation alias to_ from := (mem_pair to_ from (t_alias T)).
Abbreviation any_usage x := (mem_node x (t_any_usage T)).
Abbreviation any_source x := (mem_node x (t_any_source T)).
Abbreviation has_usage x := (mem_node x (t_has_usage T)).
Abbreviation has_source x := (mem_node x (t_has_source T)).
Abbreviation usages x y := (mem_pair x y (t_usages T)).
Abbreviation sources x y := (mem_pair x y (t_sources T)).
Abbreviation reading_field relation x :=
  (mem_field_node relation x (t_reading_field T)).
Abbreviation escaping_field relation x :=
  (mem_field_node relation x (t_escaping_field T)).
Abbreviation field_of_constructor_is_used base relation :=
  (mem_node_field base relation (t_field_used T)).

(* ======================================================================== *)
(* alias_rules                                                              *)
(* ======================================================================== *)

Definition check_alias_input : bool :=
  forallb (fun '(to_, from) => alias to_ from) (f_alias F).

Definition check_alias_propagate : bool :=
  forallb
    (fun '(if_used, to_, from) => negb (any_usage if_used) || alias to_ from)
    (f_propagate F).

Definition check_alias_if_any_source : bool :=
  forallb
    (fun '(if_any_source, to_, from) =>
       negb (any_source if_any_source) || alias to_ from)
    (f_alias_if_any_source F).

Definition check_alias_constructor_usage : bool :=
  forallb
    (fun '(base, relation, from) =>
       any_usage base
       || forallb
            (fun '(base', base_use) =>
               negb (node_eqb base base')
               || forallb
                    (fun '(to_, relation', base_use') =>
                       negb (field_eqb relation relation'
                             && node_eqb base_use base_use')
                       || alias to_ from)
                    (f_accessor F))
            (t_usages T))
    (f_constructor F).

Definition check_alias_local_constructor_usage : bool :=
  forallb
    (fun '(base, relation, from) =>
       negb (is_local relation)
       || forallb
            (fun '(base', base_use) =>
               negb (node_eqb base base')
               || forallb
                    (fun '(to_, relation', base_use') =>
                       negb (field_eqb relation relation'
                             && node_eqb base_use base_use')
                       || alias to_ from)
                    (f_accessor F))
            (t_usages T))
    (f_constructor F).

Definition check_alias_parameter_usage : bool :=
  forallb
    (fun '(base, relation, to_) =>
       any_usage base
       || forallb
            (fun '(base', base_use) =>
               negb (node_eqb base base')
               || forallb
                    (fun '(from, relation', base_use') =>
                       negb (Nat.eqb relation relation'
                             && node_eqb base_use base_use')
                       || alias to_ from)
                    (f_argument F))
            (t_usages T))
    (f_parameter F).

Definition check_alias_accessor_source : bool :=
  forallb
    (fun '(to_, relation, base) =>
       any_source base
       || forallb
            (fun '(base', base_source) =>
               negb (node_eqb base base')
               || forallb
                    (fun '(base_source', relation', from) =>
                       negb (node_eqb base_source base_source'
                             && field_eqb relation relation')
                       || alias to_ from)
                    (f_constructor F))
            (t_sources T))
    (f_accessor F).

Definition check_alias_local_accessor_source : bool :=
  forallb
    (fun '(to_, relation, base) =>
       negb (is_local relation)
       || forallb
            (fun '(base', base_source) =>
               negb (node_eqb base base')
               || forallb
                    (fun '(base_source', relation', from) =>
                       negb (node_eqb base_source base_source'
                             && field_eqb relation relation')
                       || alias to_ from)
                    (f_constructor F))
            (t_sources T))
    (f_accessor F).

Definition check_alias_escaping_reading : bool :=
  forallb
    (fun '(relation, from) =>
       forallb
         (fun '(relation', to_) =>
            negb (field_eqb relation relation') || alias to_ from)
         (t_reading_field T))
    (t_escaping_field T).

Definition check_alias_argument_source : bool :=
  forallb
    (fun '(from, relation, base) =>
       any_source base
       || forallb
            (fun '(base', base_source) =>
               negb (node_eqb base base')
               || forallb
                    (fun '(base_source', relation', to_) =>
                       negb (node_eqb base_source base_source'
                             && Nat.eqb relation relation')
                       || alias to_ from)
                    (f_parameter F))
            (t_sources T))
    (f_argument F).

Definition check_alias_rules : bool :=
  check_alias_input && check_alias_propagate && check_alias_if_any_source
  && check_alias_constructor_usage && check_alias_local_constructor_usage
  && check_alias_parameter_usage && check_alias_accessor_source
  && check_alias_local_accessor_source && check_alias_escaping_reading
  && check_alias_argument_source.

(* ======================================================================== *)
(* has_usage_rules                                                          *)
(* ======================================================================== *)

Definition check_has_usage_any : bool :=
  forallb (fun x => has_usage x) (t_any_usage T).

Definition check_has_usage_alias : bool :=
  forallb
    (fun '(to_, from) => negb (has_usage to_) || has_usage from)
    (t_alias T).

Definition check_has_usage_accessor : bool :=
  forallb
    (fun '(to_, _, base) => negb (has_usage to_) || has_usage base)
    (f_accessor F).

Definition check_has_usage_argument : bool :=
  forallb
    (fun '(from, _, base) => negb (has_source from) || has_usage base)
    (f_argument F).

Definition check_has_usage_rules : bool :=
  check_has_usage_any && check_has_usage_alias && check_has_usage_accessor
  && check_has_usage_argument.

(* ======================================================================== *)
(* has_source_rules                                                         *)
(* ======================================================================== *)

Definition check_has_source_any : bool :=
  forallb (fun x => has_source x) (t_any_source T).

Definition check_has_source_alias : bool :=
  forallb
    (fun '(to_, from) => negb (has_source from) || has_source to_)
    (t_alias T).

Definition check_has_source_constructor : bool :=
  forallb
    (fun '(base, _, from) => negb (has_source from) || has_source base)
    (f_constructor F).

Definition check_has_source_function_slot : bool :=
  forallb
    (fun '(base, relation, _) =>
       negb (is_function_slot relation) || has_source base)
    (f_constructor F).

Definition check_has_source_parameter : bool :=
  forallb
    (fun '(base, _, to_) => negb (has_usage to_) || has_source base)
    (f_parameter F).

Definition check_has_source_rules : bool :=
  check_has_source_any && check_has_source_alias
  && check_has_source_constructor && check_has_source_function_slot
  && check_has_source_parameter.

(* ======================================================================== *)
(* any_usage_rules, and the input [any_usage]                               *)
(* ======================================================================== *)

Definition check_any_usage_use : bool :=
  forallb
    (fun '(to_, from) => negb (has_usage to_) || any_usage from)
    (f_use F).

Definition check_any_usage_alias : bool :=
  forallb
    (fun '(to_, from) => negb (any_usage to_) || any_usage from)
    (t_alias T).

Definition check_any_usage_constructor : bool :=
  forallb
    (fun '(base, relation, from) =>
       negb (any_usage base) || is_local relation || any_usage from)
    (f_constructor F).

Definition check_any_usage_argument : bool :=
  forallb
    (fun '(from, _, base) => negb (any_source base) || any_usage from)
    (f_argument F).

Definition check_any_usage_input : bool :=
  forallb (fun x => any_usage x) (f_any_usage F).

Definition check_any_usage_rules : bool :=
  check_any_usage_use && check_any_usage_alias && check_any_usage_constructor
  && check_any_usage_argument && check_any_usage_input.

(* ======================================================================== *)
(* any_source_rules, and the input [any_source]                             *)
(* ======================================================================== *)

Definition check_any_source_input : bool :=
  forallb (fun x => any_source x) (f_any_source F).

Definition check_any_source_zero_alloc : bool :=
  forallb (fun x => any_source x) (f_zero_alloc_source F).

Definition check_any_source_if_used : bool :=
  forallb
    (fun '(if_used, x) => negb (any_usage if_used) || any_source x)
    (f_any_source_if_used F).

Definition check_any_source_alias : bool :=
  forallb
    (fun '(to_, from) => negb (any_source from) || any_source to_)
    (t_alias T).

Definition check_any_source_parameter : bool :=
  forallb
    (fun '(base, _, to_) => negb (any_usage base) || any_source to_)
    (f_parameter F).

Definition check_any_source_accessor : bool :=
  forallb
    (fun '(to_, relation, base) =>
       negb (any_source base) || is_local relation || any_source to_)
    (f_accessor F).

Definition check_any_source_use : bool :=
  forallb
    (fun '(to_, from) => negb (has_source from) || any_source to_)
    (f_use F).

Definition check_any_source_rules : bool :=
  check_any_source_input && check_any_source_zero_alloc
  && check_any_source_if_used && check_any_source_alias
  && check_any_source_parameter && check_any_source_accessor
  && check_any_source_use.

(* ======================================================================== *)
(* usages_rules, with the guard of [nontop_usages]                          *)
(* ======================================================================== *)

Definition check_usages_accessor : bool :=
  forallb
    (fun '(_, _, base) => any_usage base || usages base base)
    (f_accessor F).

Definition check_usages_argument : bool :=
  forallb
    (fun '(_, _, base) => any_usage base || usages base base)
    (f_argument F).

Definition check_usages_alias : bool :=
  forallb
    (fun '(to_, usage) =>
       any_usage to_
       || forallb
            (fun '(to_', from) =>
               negb (node_eqb to_ to_') || any_usage from
               || usages from usage)
            (t_alias T))
    (t_usages T).

Definition check_usages_rules : bool :=
  check_usages_accessor && check_usages_argument && check_usages_alias.

(* ======================================================================== *)
(* sources_rules, with the guard of [nontop_sources]                        *)
(* ======================================================================== *)

Definition check_sources_constructor : bool :=
  forallb
    (fun '(base, _, _) => any_source base || sources base base)
    (f_constructor F).

Definition check_sources_parameter : bool :=
  forallb
    (fun '(base, _, _) => any_source base || sources base base)
    (f_parameter F).

Definition check_sources_alias : bool :=
  forallb
    (fun '(from, source) =>
       any_source from
       || forallb
            (fun '(to_, from') =>
               negb (node_eqb from from') || any_source to_
               || sources to_ source)
            (t_alias T))
    (t_sources T).

Definition check_sources_rules : bool :=
  check_sources_constructor && check_sources_parameter
  && check_sources_alias.

(* ======================================================================== *)
(* local_rules                                                              *)
(* ======================================================================== *)

Definition check_escaping_field_rule : bool :=
  forallb
    (fun '(base, relation, from) =>
       negb (any_usage base && is_local relation)
       || escaping_field relation from)
    (f_constructor F).

Definition check_reading_field_rule : bool :=
  forallb
    (fun '(to_, relation, base) =>
       negb (any_source base && is_local relation)
       || reading_field relation to_)
    (f_accessor F).

Definition check_local_rules : bool :=
  check_escaping_field_rule && check_reading_field_rule.

(* ======================================================================== *)
(* post_processing_rules: [field_of_constructor_is_used]                    *)
(* ======================================================================== *)

Definition check_field_used_any_usage : bool :=
  forallb
    (fun '(base, relation, _) =>
       negb (any_usage base) || is_local relation
       || field_of_constructor_is_used base relation)
    (f_constructor F).

Definition check_field_used_usage : bool :=
  forallb
    (fun '(base, relation, _) =>
       forallb
         (fun '(base', usage) =>
            negb (node_eqb base base')
            || forallb
                 (fun '(v, relation', usage') =>
                    negb (field_eqb relation relation'
                          && node_eqb usage usage')
                    || negb (has_usage v)
                    || field_of_constructor_is_used base relation)
                 (f_accessor F))
         (t_usages T))
    (f_constructor F).

Definition check_field_used_reading : bool :=
  forallb
    (fun '(base, relation, _) =>
       negb (any_usage base)
       || forallb
            (fun '(relation', x) =>
               negb (field_eqb relation relation') || negb (has_usage x)
               || field_of_constructor_is_used base relation)
            (t_reading_field T))
    (f_constructor F).

Definition check_field_used_local_source : bool :=
  forallb
    (fun '(base, relation, _) =>
       negb (is_local relation && any_usage base)
       || forallb
            (fun '(usage, base') =>
               negb (node_eqb base base')
               || forallb
                    (fun '(v, relation', usage') =>
                       negb (field_eqb relation relation'
                             && node_eqb usage usage')
                       || negb (has_usage v)
                       || field_of_constructor_is_used base relation)
                    (f_accessor F))
            (t_sources T))
    (f_constructor F).

Definition check_post_processing_rules : bool :=
  check_field_used_any_usage && check_field_used_usage
  && check_field_used_reading && check_field_used_local_source.

(* ======================================================================== *)
(* The closure of the solution                                              *)
(* ======================================================================== *)

Definition check_closed : bool :=
  check_alias_rules && check_has_usage_rules && check_has_source_rules
  && check_any_usage_rules && check_any_source_rules && check_usages_rules
  && check_sources_rules && check_local_rules && check_post_processing_rules.

(* ======================================================================== *)
(* [field_of_constructor_is_used] holds only where a rule gives it          *)
(* ======================================================================== *)

(* Whether each rule of [post_processing_rules] gives
   [field_of_constructor_is_used base relation]. The rules whose last
   premise is [any_usage v] are left out, as in
   FlambdaReaperGraphSpec.field_used_exact: on tables that pass
   [check_closed], [any_usage v] gives [has_usage v]. *)
Definition derived_by_field_used_any_usage (base : node) (relation : field)
    : bool :=
  existsb
    (fun '(base', relation', _) =>
       node_eqb base base' && field_eqb relation relation'
       && any_usage base && negb (is_local relation))
    (f_constructor F).

Definition derived_by_field_used_usage (base : node) (relation : field)
    : bool :=
  existsb
    (fun '(base', relation', _) =>
       node_eqb base base' && field_eqb relation relation'
       && existsb
            (fun '(base'', usage) =>
               node_eqb base base''
               && existsb
                    (fun '(v, relation'', usage') =>
                       field_eqb relation relation''
                       && node_eqb usage usage' && has_usage v)
                    (f_accessor F))
            (t_usages T))
    (f_constructor F).

Definition derived_by_field_used_reading (base : node) (relation : field)
    : bool :=
  existsb
    (fun '(base', relation', _) =>
       node_eqb base base' && field_eqb relation relation'
       && any_usage base
       && existsb
            (fun '(relation'', x) =>
               field_eqb relation relation'' && has_usage x)
            (t_reading_field T))
    (f_constructor F).

Definition derived_by_field_used_local_source (base : node)
    (relation : field) : bool :=
  existsb
    (fun '(base', relation', _) =>
       node_eqb base base' && field_eqb relation relation'
       && is_local relation && any_usage base
       && existsb
            (fun '(usage, base'') =>
               node_eqb base base''
               && existsb
                    (fun '(v, relation'', usage') =>
                       field_eqb relation relation''
                       && node_eqb usage usage' && has_usage v)
                    (f_accessor F))
            (t_sources T))
    (f_constructor F).

Definition check_field_used_exact : bool :=
  forallb
    (fun '(base, relation) =>
       derived_by_field_used_any_usage base relation
       || derived_by_field_used_usage base relation
       || derived_by_field_used_reading base relation
       || derived_by_field_used_local_source base relation)
    (t_field_used T).

End Checks.
