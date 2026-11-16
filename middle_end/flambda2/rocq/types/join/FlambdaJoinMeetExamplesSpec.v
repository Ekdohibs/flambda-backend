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


(* Examples of the n-way join of typing environments: what the join
   that Simplify calls answers on small inputs, with the meet, the
   join of types and the join of environments of the model put
   together ([model_join], FlambdaJoinMeet.v).

   To be reviewed: how the inputs are made, and each statement,
   against what the compiler's [Join_levels.cut_and_n_way_join]
   answers at a continuation whose parameter is given the types of
   the example at its uses.

   Why they are here. What is proved of the join says that what it
   adds to the target is true. A join that added nothing would have
   that too. The examples say what the model computes, with nothing
   stood in for: the same functions as in the statements, and
   environments made with the model's own [Typing_env.add_equation].
   An answer [None] on one of these inputs would be a fatal error of
   the compiler, too little fuel, or the state at which the model's
   meet stops and the compiler does not (FlambdaJoinMeet.v says
   which): there is none.

   What they are not. No answer here was read off a run of the
   compiler: each is what the model computes, and was compared with
   the compiler's code by reading.

   They have since been compared with the compiler by running, by the
   workstream of the differential tests of the types library, for
   every section on the join that Simplify calls but the one with
   three joins in a row. The inputs were made in the compiler
   as they are here ([increment_scope], a definition and
   [Meet_env.add_equation] for each variable, then for [p]), and
   joined both by [Join_env.cut_and_n_way_join_with_analysis] and by
   [Join_levels.cut_and_n_way_join] with the n-way join chosen. The
   twelve agree with the statements: the type of [p], whether the
   target is bottom, the type of [a], and the two maps of
   definitions, sets compared as sets. What the compiler shows and
   the statements do not: where a variable is made for [Is_int] or
   [Get_tag], it has a type in the target, the numbers 0 and 1 with
   the relation on [p].

   The section with three joins in a row is of the shape of the
   program of discrepancy T5-10. Its answers were compared with what
   the compiler prints for the three joins of that program (run by
   the workstream that demonstrated the discrepancy, with the n-way
   join and the types of the joins printed): on the variables that
   the example keeps, the equations are the same.

   The last two sections are not about the join that Simplify calls,
   and were not compared with the compiler: a meet that calls the
   join, and a join whose last phase runs a meet that calls the
   join. Each says what it is for.

   The inputs. The source environment knows three variables of kind
   value that may occur in terms: [y], [z] and the parameter [p]. A
   use is the source one scope further, in which some variables may
   have been defined and given types, and in which [p] has been given
   a type, all by [Typing_env.add_equation]. The uses are joined
   after the scope of the source. What a statement shows of the
   answer: the type that the target has for [p]; whether the target
   is bottom; the two maps of definitions of the analysis; and the
   type that the target has for [a], a variable that some uses define
   and the source does not.

   The choices are the compiler's, the depth of the join is the
   compiler's default, 5, and the fuel is more than these inputs
   need. A new variable is one greater than the names to avoid, which
   are [y], [z], [p] and [a]: so the first variable that a join makes
   here is 5. The sets of numbers of the model are lists, and the
   order of the numbers in an answer is the one that the model makes;
   it has no meaning.

   The proofs are by computation, in FlambdaJoinMeetExamplesProofs.v,
   which needs no review. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaTypingEnv FlambdaJoinEnv FlambdaJoinSpec FlambdaJoinEnvSpec
  FlambdaJoinTypesExamplesSpec FlambdaMeetGrammar FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaJoinMeet.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The inputs                                                               *)
(* ======================================================================== *)

Definition example_y : variable := 1%positive.
Definition example_z : variable := 2%positive.
Definition example_p : variable := 3%positive.
Definition example_a : variable := 4%positive.

Definition example_fuel : fuels :=
  {| fuel_of_env := 20; fuel_of_types := 20; fuel_of_meet := 20 |}.

Definition example_source_env : Typing_env.t :=
  Typing_env.add_variable_definition
    (Typing_env.add_variable_definition
       (Typing_env.add_variable_definition Typing_env.create example_y Value
          Name_mode.normal)
       example_z Value Name_mode.normal)
    example_p Value Name_mode.normal.

Definition example_add_equation : add_equation_function :=
  model_add_equation Join_env.compiler_choices compiler_choices 5
    example_fuel.

(* An environment in which variables have been defined, each at a name
   mode and with a type. *)
Fixpoint example_with_variables (tenv : Typing_env.t)
    (variables : list (variable * Name_mode.t * flambda_type))
    : option Typing_env.t :=
  match variables with
  | [] => Some tenv
  | (x, name_mode, ty) :: variables =>
    let* tenv :=
      example_add_equation
        (Typing_env.add_variable_definition tenv x (kind ty) name_mode)
        (Name_var x) ty
    in
    example_with_variables tenv variables
  end.

(* A use: the source one scope further, with the variables given, and
   the type given for [p]. *)
Definition example_use
    (variables : list (variable * Name_mode.t * flambda_type))
    (ty : flambda_type) : option Typing_env.t :=
  let* tenv :=
    example_with_variables (Typing_env.increment_scope example_source_env)
      variables
  in
  example_add_equation tenv (Name_var example_p) ty.

(* A use that is bottom. *)
Definition example_bottom_use : option Typing_env.t :=
  Some (Typing_env.make_bottom (Typing_env.increment_scope example_source_env)).

Fixpoint example_all (uses : list (option Typing_env.t))
    : option (list Typing_env.t) :=
  match uses with
  | [] => Some []
  | use :: uses =>
    let* use := use in
    let* uses := example_all uses in
    Some (use :: uses)
  end.

(* What is shown of an answer. *)
Record example_answer : Type := Example_answer {
  type_of_p : option flambda_type;
  target_is_bottom : bool;
  definitions : finite_map (Simples_in_joined_envs.t * flambda_kind);
  definitions_at_normal_mode
    : finite_map (Simples_in_joined_envs.t * flambda_kind);
  type_of_a : option flambda_type
}.

(* The join of uses. *)
Definition example_join_uses (uses : list (option Typing_env.t))
    : option example_answer :=
  let* uses := example_all uses in
  let* '(target, analysis) :=
    model_join Join_env.compiler_choices compiler_choices 5 example_fuel
      [example_y; example_z; example_p; example_a]
      {| source := example_source_env;
         cut_after := Typing_env.current_scope example_source_env;
         uses := map (fun use => (tt, use)) uses |}
  in
  Some
    {| type_of_p := Typing_env.find target (Name_var example_p) None;
       target_is_bottom := Typing_env.is_bottom target;
       definitions := Analysis.definitions_in_joined_envs analysis;
       definitions_at_normal_mode :=
         Analysis.canonical_definitions_at_normal_mode analysis;
       type_of_a :=
         if Typing_env.mem None target (Name_var example_a)
         then Typing_env.find target (Name_var example_a) None
         else None |}.

(* The join of two uses that define no variable. *)
Definition example_join_types (ty1 ty2 : flambda_type)
    : option example_answer :=
  example_join_uses [example_use [] ty1; example_use [] ty2].

(* An answer in which [p] has a type, the target is not bottom, the two
   maps of definitions are the same, and the target does not know
   [a]. *)
Definition example_answer_of (ty : flambda_type)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    : option example_answer :=
  Some
    {| type_of_p := Some ty;
       target_is_bottom := false;
       definitions := definitions;
       definitions_at_normal_mode := definitions;
       type_of_a := None |}.

(* The definition of a variable at two uses. *)
Definition example_definition (x : variable) (indices : nat * nat)
    (simples : simple * simple) (kind : flambda_kind)
    : finite_map (Simples_in_joined_envs.t * flambda_kind) :=
  [(x,
    ([(fst indices, fst simples); (snd indices, snd simples)], kind))].

(* ======================================================================== *)
(* Constants                                                                *)
(* ======================================================================== *)

(* [p] is 1 at a use and 2 at the other. The join of types makes a
   variable for the two constants, of which [p] is an alias in the
   target; so the variable is replaced by [p], which has its type, the
   set of the two, and its definition: [p] is the constant 1 at the
   first use and 2 at the second. *)
Definition example_join_env_two_constants_statement : Prop :=
  example_join_types (this_tagged_immediate 1) (this_tagged_immediate 2)
  = example_answer_of (these_tagged_immediates [1; 2])
      (example_definition example_p (0%nat, 1%nat)
         (Const (Tagged_immediate 1), Const (Tagged_immediate 2)) Value).

(* The same constant at both uses: [p] is that constant, and nothing
   is defined. *)
Definition example_join_env_one_constant_statement : Prop :=
  example_join_types (this_tagged_immediate 1) (this_tagged_immediate 1)
  = example_answer_of (this_tagged_immediate 1) [].

(* A set of numbers at a use and a constant at the other: the union,
   and nothing is defined. *)
Definition example_join_env_set_and_constant_statement : Prop :=
  example_join_types (these_tagged_immediates [1; 2])
    (this_tagged_immediate 3)
  = example_answer_of (these_tagged_immediates [3; 1; 2]) [].

(* ======================================================================== *)
(* Aliases                                                                  *)
(* ======================================================================== *)

(* [p] is [y] at both uses: it is [y] in the target. *)
Definition example_join_env_one_alias_statement : Prop :=
  example_join_types (alias_type_of Value (Var example_y))
    (alias_type_of Value (Var example_y))
  = example_answer_of (alias_type_of Value (Var example_y)) [].

(* [p] is [y] at a use and [z] at the other: nothing is known of it
   in the target (nothing is known of [y] nor of [z]), and the
   analysis says what it is at each use. *)
Definition example_join_env_two_aliases_statement : Prop :=
  example_join_types (alias_type_of Value (Var example_y))
    (alias_type_of Value (Var example_z))
  = example_answer_of (unknown Value)
      (example_definition example_p (0%nat, 1%nat)
         (Var example_y, Var example_z) Value).

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

(* Two pairs of constants: a pair of sets. The two variables that the
   join of types makes for the fields are mentioned once each, and are
   replaced by their types. *)
Definition example_join_env_pairs_statement : Prop :=
  example_join_types
    (example_join_block 0 [this_tagged_immediate 1; this_tagged_immediate 2])
    (example_join_block 0 [this_tagged_immediate 3; this_tagged_immediate 4])
  = example_answer_of
      (example_join_block 0
         [these_tagged_immediates [1; 3]; these_tagged_immediates [2; 4]])
      [].

(* An immediate at a use and a block at the other, as for a value of
   an option type: a variant with both, and a variable made for the
   result of [Is_int], which is 1 at the first use and 0 at the
   second. *)
Definition example_join_env_immediate_and_block_statement : Prop :=
  example_join_types (this_tagged_immediate 0)
    (example_join_block 0 [alias_type_of Value (Var example_y)])
  = example_answer_of
      (example_join_variant (Some 5%positive) (this_naked_immediate 0) None
         [(0, [alias_type_of Value (Var example_y)])])
      (example_definition 5%positive (1%nat, 0%nat)
         (Const (Naked_immediate 0), Const (Naked_immediate 1))
         (Naked_number Kind_naked_immediate)).

(* Blocks of two tags: a variant with the two cases, and a variable
   made for the result of [Get_tag], which is 0 at the first use and 1
   at the second. *)
Definition example_join_env_two_tags_statement : Prop :=
  example_join_types (example_join_block 0 [this_tagged_immediate 1])
    (example_join_block 1 [this_tagged_immediate 2])
  = example_answer_of
      (example_join_variant None (bottom (Naked_number Kind_naked_immediate))
         (Some 5%positive)
         [(0, [this_tagged_immediate 1]); (1, [this_tagged_immediate 2])])
      (example_definition 5%positive (1%nat, 0%nat)
         (Const (Naked_immediate 1), Const (Naked_immediate 0))
         (Naked_number Kind_naked_immediate)).

(* ======================================================================== *)
(* Variables of the uses                                                    *)
(* ======================================================================== *)

(* The field of the block is a variable that the first use defines,
   one of two numbers, and a constant at the second use: the field is
   one of the three in the target, which does not know the variable. *)
Definition example_join_env_variable_of_one_use_statement : Prop :=
  example_join_uses
    [example_use
       [(example_a, Name_mode.normal, these_tagged_immediates [1; 2])]
       (example_join_block 0 [alias_type_of Value (Var example_a)]);
     example_use [] (example_join_block 0 [this_tagged_immediate 7])]
  = example_answer_of
      (example_join_block 0 [these_tagged_immediates [7; 1; 2]]) [].

(* The field is a variable that both uses define, which exists only in
   types: the target has the variable, with the type that it has at
   the uses, and the field is that variable. *)
Definition example_join_env_variable_of_both_uses_statement : Prop :=
  example_join_uses
    [example_use
       [(example_a, Name_mode.in_types, these_tagged_immediates [1; 2])]
       (example_join_block 0 [alias_type_of Value (Var example_a)]);
     example_use
       [(example_a, Name_mode.in_types, these_tagged_immediates [1; 2])]
       (example_join_block 0 [alias_type_of Value (Var example_a)])]
  = Some
      {| type_of_p :=
           Some (example_join_block 0 [alias_type_of Value (Var example_a)]);
         target_is_bottom := false;
         definitions := [];
         definitions_at_normal_mode := [];
         type_of_a := Some (these_tagged_immediates [1; 2]) |}.

(* ======================================================================== *)
(* Uses that are bottom                                                     *)
(* ======================================================================== *)

(* A use that is bottom is left out: the definitions are at the first
   and the third. *)
Definition example_join_env_bottom_use_statement : Prop :=
  example_join_uses
    [example_use [] (this_tagged_immediate 1); example_bottom_use;
     example_use [] (this_tagged_immediate 2)]
  = example_answer_of (these_tagged_immediates [1; 2])
      (example_definition example_p (0%nat, 2%nat)
         (Const (Tagged_immediate 1), Const (Tagged_immediate 2)) Value).

(* No use that is not bottom: the target is bottom. *)
Definition example_join_env_no_use_statement : Prop :=
  example_join_uses [example_bottom_use]
  = Some
      {| type_of_p := Some (bottom Value);
         target_is_bottom := true;
         definitions := [];
         definitions_at_normal_mode := [];
         type_of_a := None |}.

(* ======================================================================== *)
(* A relation recovered on a variable that has no value at a use            *)
(* ======================================================================== *)

(* Three joins, one after another, as Simplify makes them for a
   function of this shape, where [t0] is a naked immediate and [n] a
   value, both known before:

     if c then
       (switch t0 with
        | 0 -> b0 = block of tag 0 (n); kx (b0)
        | 1 -> b1 = block of tag 1 (n); kx (b1))
       where kx (x) = pa = block of tag 0 (x); k0 (pa)
     else pb = block of tag 1 (n); k0 (pb)
     where k0 (p) =
       if g then k2 (p)
       else (switch t0 with
             | 0 -> z = block of tag 0 (n);
                    pd = block of tag 0 (z); k2 (pd))
       where k2 (s) = switch t0 with 0 -> .. | 1 -> .. | 2 -> ..

   It is the shape of the program of discrepancy T5-10 (which has two
   more parameters, left out here), for which the compiler prints the
   types after each of its three joins, and deletes the last arm,
   which a run through [pb] and [k2 (p)] with [t0] equal to 2
   reaches. The model answers the same equations on these variables,
   and the same bottom:
   - at [kx], the tag of [x] is 0 at a use and 1 at the other, as
     [t0] is, so the join names it [t0], and records that [t0] is the
     tag of [x];
   - at [k0], [x] is a variable of the first use only; the join brings
     it into the target under its own name, with the type that it has
     at that use, which says that its tag is [t0]. On a run through
     the second use, [x] has no value and [t0] is any number;
   - at [k2], the field of the block is [x] at a use and [z] at the
     other; the join makes a variable for it, whose tag is [t0] at
     both ([z] is made where [t0] is 0), and records that [t0] is the
     tag of that variable. Its test for recording it is that every
     use has a type for the variable, and the first has one for [x];
   - in the arm where [t0] is 2, the meet reads the relation: the
     variable would be a block of tag 2, and its type has the tags 0
     and 1. The environment of the arm is bottom.
   This does not contradict [model_join_true_statement]: what it
   concludes is about the runs on which the scrutinees of the
   relations that the join adds have values
   ([relation_scrutinees_have_values]). At [k2], on the run through
   the second use of [k0] and the first of [k2], the new variable
   stands for [x], which has no value, while [t0] has one. (At [k0]
   nothing fails: the second use does not know [x], and nothing is
   asked or said of a variable brought in at a use that does not know
   it.) The statements here are computations; they prove nothing of
   runs. *)

Module Recovered.

  Definition n : variable := 1%positive.
  Definition t0 : variable := 2%positive.
  Definition x : variable := 3%positive.
  Definition p : variable := 4%positive.
  Definition s : variable := 5%positive.
  Definition b0 : variable := 6%positive.
  Definition b1 : variable := 7%positive.
  Definition pa : variable := 8%positive.
  Definition pb : variable := 9%positive.
  Definition z : variable := 10%positive.
  Definition pd : variable := 11%positive.

  (* The variables that the second join and the third make. *)
  Definition tag_of_p : variable := 12%positive.
  Definition tag_of_s : variable := 13%positive.
  Definition field_of_s : variable := 14%positive.

  Definition define (tenv : Typing_env.t) (var : variable)
      (kind : flambda_kind) : Typing_env.t :=
    Typing_env.add_variable_definition tenv var kind Name_mode.normal.

  Definition alias (var : variable) : flambda_type :=
    alias_type_of Value (Var var).

  (* A join after the scope of the source; [made] is the variables
     that the joins before have made. *)
  Definition join (made : list variable) (source : Typing_env.t)
      (uses : list Typing_env.t) : option Typing_env.t :=
    let* '(target, _) :=
      model_join Join_env.compiler_choices compiler_choices 5 example_fuel
        ([n; t0; x; p; s; b0; b1; pa; pb; z; pd] ++ made)
        {| source := source;
           cut_after := Typing_env.current_scope source;
           uses := map (fun use => (tt, use)) uses |}
    in
    Some target.

  (* A variable that may occur in terms, defined with a type. *)
  Definition with_variable (tenv : Typing_env.t) (var : variable)
      (ty : flambda_type) : option Typing_env.t :=
    example_with_variables tenv [(var, Name_mode.normal, ty)].

  (* Where [k0] is defined. *)
  Definition at_k0 : Typing_env.t :=
    define
      (define (define Typing_env.create n Value) t0
         (Naked_number Kind_naked_immediate))
      p Value.

  (* Where [kx] is defined. *)
  Definition at_kx : Typing_env.t :=
    define (Typing_env.increment_scope at_k0) x Value.

  (* The use of [kx] in the arm of [t0] for a number. *)
  Definition use_of_kx (tag : Z) (block : variable) : option Typing_env.t :=
    let* tenv :=
      example_add_equation (Typing_env.increment_scope at_kx) (Name_var t0)
        (this_naked_immediate tag)
    in
    let* tenv := with_variable tenv block (example_join_block tag [alias n]) in
    example_add_equation tenv (Name_var x) (alias block).

  Definition after_kx : option Typing_env.t :=
    let* use0 := use_of_kx 0 b0 in
    let* use1 := use_of_kx 1 b1 in
    join [] at_kx [use0; use1].

  (* The two uses of [k0]. *)
  Definition first_use_of_k0 : option Typing_env.t :=
    let* tenv := after_kx in
    let* tenv := with_variable tenv pa (example_join_block 0 [alias x]) in
    example_add_equation tenv (Name_var p) (alias pa).

  Definition second_use_of_k0 : option Typing_env.t :=
    let* tenv :=
      with_variable (Typing_env.increment_scope at_k0) pb
        (example_join_block 1 [alias n])
    in
    example_add_equation tenv (Name_var p) (alias pb).

  Definition after_k0 : option Typing_env.t :=
    let* use0 := first_use_of_k0 in
    let* use1 := second_use_of_k0 in
    join [] at_k0 [use0; use1].

  (* Where [k2] is defined, and its two uses. *)
  Definition at_k2 : option Typing_env.t :=
    let* tenv := after_k0 in
    Some (define tenv s Value).

  Definition first_use_of_k2 : option Typing_env.t :=
    let* tenv := at_k2 in
    example_add_equation (Typing_env.increment_scope tenv) (Name_var s)
      (alias p).

  Definition second_use_of_k2 : option Typing_env.t :=
    let* tenv := at_k2 in
    let* tenv :=
      example_add_equation (Typing_env.increment_scope tenv) (Name_var t0)
        (this_naked_immediate 0)
    in
    let* tenv := with_variable tenv z (example_join_block 0 [alias n]) in
    let* tenv := with_variable tenv pd (example_join_block 0 [alias z]) in
    example_add_equation tenv (Name_var s) (alias pd).

  Definition after_k2 : option Typing_env.t :=
    let* source := at_k2 in
    let* use0 := first_use_of_k2 in
    let* use1 := second_use_of_k2 in
    join [tag_of_p] source [use0; use1].

  (* The arm of the last switch where [t0] is 2. *)
  Definition arm_two : option Typing_env.t :=
    let* tenv := after_k2 in
    example_add_equation tenv (Name_var t0) (this_naked_immediate 2).

  (* The types that an environment has for some variables. *)
  Definition types_of (tenv : option Typing_env.t) (vars : list variable)
      : option (list (option flambda_type)) :=
    let* tenv := tenv in
    Some (map (fun var => Typing_env.find tenv (Name_var var) None) vars).

  (* A block of tag 0 or 1 with one field: the variable given if the
     tag is 0, [n] if it is 1; its tag is held by a variable. *)
  Definition block_with_tag_in (tag : variable) (field : variable)
      : flambda_type :=
    example_join_variant None (bottom (Naked_number Kind_naked_immediate))
      (Some tag) [(0, [alias field]); (1, [alias n])].

  (* A naked immediate that is the tag of a variable. *)
  Definition tag_of (numbers : Or_unknown.t (list Z)) (block : variable)
      : flambda_type :=
    Type_naked_immediate
      (Ok
         (No_alias
            (Head_of_kind_naked_immediate numbers
               [(Relation.Get_tag, Name_var block)]))).

End Recovered.

(* After the join at [kx]: the tag of [x] is [t0], and [t0] is 0 or 1
   and the tag of [x]. *)
Definition example_recovered_first_join_statement : Prop :=
  Recovered.types_of Recovered.after_kx [Recovered.x; Recovered.t0]
  = Some
      [Some (Recovered.block_with_tag_in Recovered.t0 Recovered.n);
       Some (Recovered.tag_of (Or_unknown.Known [0; 1]) Recovered.x)].

(* After the join at [k0]: the field of [p] is [x] if its tag is 0;
   [x] has the type that it had; nothing is known of [t0]. *)
Definition example_recovered_second_join_statement : Prop :=
  Recovered.types_of Recovered.after_k0
    [Recovered.p; Recovered.x; Recovered.t0; Recovered.tag_of_p]
  = Some
      [Some (Recovered.block_with_tag_in Recovered.tag_of_p Recovered.x);
       Some (Recovered.block_with_tag_in Recovered.t0 Recovered.n);
       Some (unknown (Naked_number Kind_naked_immediate));
       Some (Recovered.tag_of (Or_unknown.Known [0; 1]) Recovered.p)].

(* After the join at [k2]: the field of [s] is a new variable if its
   tag is 0; the tag of that variable is [t0]; and [t0] is any number,
   and the tag of that variable. *)
Definition example_recovered_third_join_statement : Prop :=
  Recovered.types_of Recovered.after_k2
    [Recovered.s; Recovered.field_of_s; Recovered.t0; Recovered.tag_of_s]
  = Some
      [Some
         (Recovered.block_with_tag_in Recovered.tag_of_s
            Recovered.field_of_s);
       Some (Recovered.block_with_tag_in Recovered.t0 Recovered.n);
       Some (Recovered.tag_of Or_unknown.Unknown Recovered.field_of_s);
       Some (Recovered.tag_of (Or_unknown.Known [0; 1]) Recovered.s)].

(* None of the three targets is bottom, and the environment in which
   [t0] is 2 after the third is. *)
Definition example_recovered_bottom_statement : Prop :=
  option_map Typing_env.is_bottom Recovered.after_kx = Some false
  ∧ option_map Typing_env.is_bottom Recovered.after_k0 = Some false
  ∧ option_map Typing_env.is_bottom Recovered.after_k2 = Some false
  ∧ option_map Typing_env.is_bottom Recovered.arm_two = Some true.

(* ======================================================================== *)
(* A meet that joins                                                        *)
(* ======================================================================== *)

(* The meet of the model calls the join of the model
   ([Join_env.cut_and_n_way_join], through [model_join_scoped_envs])
   where the two types met may each be an immediate or a block: it
   meets the immediates in one environment and the blocks in another,
   each one scope further and with the extension that the type has
   for that alternative, and if neither is bottom it joins the two
   environments.

   Here a type says that [z] is 7 if the value is an immediate and 8
   if it is a block. The types are met in the source of the examples
   above; what is shown is the type answered and the type that the
   environment answered has for [z].

   These were not compared with the compiler. They were the first
   runs of the model's meet through the model's join, and before them
   that join took its names to avoid from the meet's environment
   alone, which has none when a meet begins: the first example then
   answered [None], the join having made a variable that the
   environment knew. It now avoids the variables that the
   environments it is given know ([Join_env.variables_of_env]). *)

(* An immediate that is 0 or 1, or a block of tag 0 with one field,
   with extensions for the two alternatives. *)
Definition example_immediate_or_block (extensions : variant_extensions)
    : flambda_type :=
  non_null_value
    (Variant None (Or_unknown.Known (these_naked_immediates [0; 1])) None
       (Or_unknown.Known
          (Row_like_for_blocks
             [(0,
               Or_unknown.Known
                 (Row_like_case [unknown Value]
                    (Row_like_index (Row_like_index_domain.Known 1%nat)
                       (Scannable Value_only))
                    empty_env_extension))]
             Or_bottom.Bottom Alloc_mode.Heap))
       extensions false).

(* [z] is one number if the value is an immediate, another if it is a
   block. *)
Definition example_z_is (when_immediate when_block : Z)
    : variant_extensions :=
  Ext
    (Env_extension
       [(Name_var example_z, this_tagged_immediate when_immediate)])
    (Env_extension [(Name_var example_z, this_tagged_immediate when_block)]).

(* The meet of two types in the source: the type answered, and the
   type of [z] in the environment answered; or bottom. *)
Definition example_meet (ty1 ty2 : flambda_type)
    : option (Or_bottom.t (flambda_type * option flambda_type)) :=
  let* answer :=
    model_meet Join_env.compiler_choices compiler_choices 5 example_fuel
      example_source_env ty1 ty2
  in
  match answer with
  | Or_bottom.Bottom => Some Or_bottom.Bottom
  | Or_bottom.Ok (ty, tenv) =>
    Some (Or_bottom.Ok (ty, Typing_env.find tenv (Name_var example_z) None))
  end.

(* Both alternatives are left: the type keeps the two extensions, and
   the environments of the two alternatives are joined, so that [z] is
   7 or 8. *)
Definition example_meet_joins_alternatives_statement : Prop :=
  example_meet (example_immediate_or_block (example_z_is 7 8))
    (example_immediate_or_block No_extensions)
  = Some
      (Or_bottom.Ok
         (example_immediate_or_block (example_z_is 7 8),
          Some (these_tagged_immediates [7; 8]))).

(* Only the immediates are left: no join, and [z] is 7. *)
Definition example_meet_immediate_alternative_statement : Prop :=
  example_meet (example_immediate_or_block (example_z_is 7 8))
    (these_tagged_immediates [0; 1])
  = Some
      (Or_bottom.Ok
         (these_tagged_immediates [0; 1], Some (this_tagged_immediate 7))).

(* Only the blocks are left: [z] is 8. *)
Definition example_meet_block_alternative_statement : Prop :=
  example_meet (example_immediate_or_block (example_z_is 7 8))
    (example_join_block 0 [unknown Value])
  = Some
      (Or_bottom.Ok
         (example_join_block 0 [unknown Value],
          Some (this_tagged_immediate 8))).

(* ======================================================================== *)
(* A join in a meet in a join, and the names that it avoids                 *)
(* ======================================================================== *)

(* The last phase of a join adds its equations to the target through
   the meet, and that meet may call the join. These are two runs of
   the join that the meet calls ([model_join_scoped_envs]) where it
   does.

   The source has [y], [z], [p] and [a], and [p] may be an immediate
   or a block: [z] is a pair of [y] and [y] if it is an immediate,
   and a pair of [a] and [a] if it is a block. Two environments are
   joined, both one scope further, in which [p] may be an immediate
   or a block and nothing more is said (they are the same; they are
   made with [Typing_env.replace_equation], with no meet). The join
   adds the type that [p] has in them to the target, where it is met
   with the type that [p] has there. Both alternatives are left, so
   the environments of the two are joined, and that join makes a
   variable for [y] and [a], of which [z] is then a pair. The
   variable is mentioned twice, so it is kept.

   What is shown: whether the environment answered knows the
   variables 5 and 6, whether each is among its names to avoid, and
   the type that it has for [z].

   These runs showed a fault of the model, which is corrected; they
   are here so that it does not come back. The compiler's new
   variables differ from every other variable by construction. The
   model's are made outside the names to avoid that the environment
   of the meet carries (FlambdaJoinEnv.v, "New names"). Before the
   correction the last phase ran its meets in the environment that
   the join was given, whose names to avoid had not what the join
   knew, and the join answered the target with the names of its own
   state, forgetting what those meets had added. So:
   - in the first run the answer knew 5, which the inner join had
     made, and 5 was not among its names to avoid: the contract that
     the meet has for its join ([join_scoped_envs_sound],
     FlambdaMeetSpec.v) says that it is;
   - in the second run, where the environments joined know 5 as a
     naked immediate and the target does not bring it in, the inner
     join made 5 again, and the answer knew 5 as a value: the same
     contract says that a variable that the answer knows is there of
     the kind that an environment joined has it at.
   Now the inner join makes 5 in the first run and 6 in the second,
   and each is among the names to avoid of the answer. (That the
   inputs have what that contract supposes is not proved; and the
   runs were not compared with the compiler, which has no names to
   avoid.) *)

Definition example_b : variable := 5%positive.

(* A block with two fields that are both the variable. *)
Definition example_pair_of (x : variable) : flambda_type :=
  example_join_block 0
    [alias_type_of Value (Var x); alias_type_of Value (Var x)].

(* The source: [a] is defined, and [p] has the type described. *)
Definition example_nested_source : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition example_source_env example_a Value
       Name_mode.normal)
    (Name_var example_p)
    (example_immediate_or_block
       (Ext
          (Env_extension [(Name_var example_z, example_pair_of example_y)])
          (Env_extension [(Name_var example_z, example_pair_of example_a)]))).

(* An environment joined: the source one scope further, with the
   variables given defined, at the mode of the variables that exist
   only in types, and [p] an immediate or a block. *)
Definition example_nested_use (variables : list (variable * flambda_kind))
    : Typing_env.t :=
  Typing_env.replace_equation
    (List.fold_left
       (fun tenv binding =>
          Typing_env.add_variable_definition tenv (fst binding)
            (snd binding) Name_mode.in_types)
       variables (Typing_env.increment_scope example_nested_source))
    (Name_var example_p) (example_immediate_or_block No_extensions).

(* What is shown of the environment answered. *)
Record example_nested_answer : Type := Example_nested_answer {
  knows_5 : bool;
  avoids_5 : bool;
  knows_6 : bool;
  avoids_6 : bool;
  type_of_z : option flambda_type
}.

(* The join of that environment with itself, as the meet calls it:
   after the scope of the source, in the environment of a meet that
   begins there. *)
Definition example_nested_join (variables : list (variable * flambda_kind))
    : option example_nested_answer :=
  let use := example_nested_use variables in
  let* result :=
    model_join_scoped_envs Join_env.compiler_choices 5 example_fuel
      (model_meet_expanded_head Join_env.compiler_choices compiler_choices 5
         example_fuel)
      (Typing_env.current_scope example_nested_source)
      (Meet_env.create example_nested_source) example_nested_source
      [use; use]
  in
  let tenv := Meet_env.typing_env result in
  Some
    {| knows_5 := Typing_env.mem None tenv (Name_var 5%positive);
       avoids_5 := among 5%positive (Meet_env.names_to_avoid result);
       knows_6 := Typing_env.mem None tenv (Name_var 6%positive);
       avoids_6 := among 6%positive (Meet_env.names_to_avoid result);
       type_of_z := Typing_env.find tenv (Name_var example_z) None |}.

(* The variable that the inner join makes is among the names to avoid
   of the answer. *)
Definition example_nested_join_names_statement : Prop :=
  example_nested_join []
  = Some
      {| knows_5 := true;
         avoids_5 := true;
         knows_6 := false;
         avoids_6 := false;
         type_of_z := Some (example_pair_of 5%positive) |}.

(* The environments joined know 5, as a naked immediate; the target
   does not bring it in. The inner join does not make it again: it
   makes 6. *)
Definition example_nested_join_avoids_statement : Prop :=
  example_nested_join [(example_b, Naked_number Kind_naked_immediate)]
  = Some
      {| knows_5 := false;
         avoids_5 := true;
         knows_6 := true;
         avoids_6 := true;
         type_of_z := Some (example_pair_of 6%positive) |}.
