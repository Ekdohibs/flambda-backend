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


(* Simplifying a [Switch] whose arms all apply one continuation: the
   [Switch] becomes that application, with arguments that are computed
   from the scrutinee by primitives.

   This file is to be reviewed: the definitions and the statements,
   against [rebuild_switch], [rebuild_arm],
   [recognize_mergeable_argument], [recognize_affine_argument],
   [rebuild_mergeable_argument] and the binder [let$] of
   middle_end/flambda2/simplify/simplify_switch_expr.ml. The first two
   slices are FlambdaSimplifySwitchSpec.v and
   FlambdaSimplifySwitchArmsSpec.v, whose notions and functions are
   used here; what is known of the primitives that have been computed
   is that of ../../rocq/FlambdaCseSpec.v; the machine is
   ../../rocq/FlambdaSemanticsSpec.v.

   What Simplify does. On the way up, when the arms that are left all
   apply the same continuation with no trap action
   ([rebuild_arm], [Mergeable]), it looks at each argument across the
   arms, in the order of their numbers
   ([recognize_mergeable_argument]). An argument can be merged where
   - it is the same in every arm ([Invariant_argument]);
   - it is, in every arm, the number of the arm as a naked immediate:
     the scrutinee is passed;
   - it is, in every arm, the number of the arm as a tagged immediate
     ([Tagged_scrutinee]): the scrutinee is tagged;
   - it is the tagged immediate 1 where the number is 0 and 0 where
     the number is 1, and there is no other arm ([Not_scrutinee]): the
     scrutinee is tagged and negated;
   - it is, in every arm, an immediate that is a slope times the number
     of the arm plus an offset, in the arithmetic of immediates, which
     wraps ([Affine_argument]): the scrutinee is multiplied and added
     to. The slope and the offset are read off the first two arms and
     checked against the others. This one is taken only where there are
     three arms or more ([enough_arms]).
   Where every argument can be merged, the [Switch] becomes the
   [Apply_cont] of the continuation to the merged arguments, under the
   [Let]s of the primitives that compute them ([rebuild_switch],
   [rebuild_mergeable_argument]). Before a primitive is bound, Simplify
   looks for a variable that already holds it: in what it knew before
   the [Switch] and in what it has bound for an earlier argument of the
   same [Switch] ([let$], [find_cse_simple]). So
   [switch i with 0 -> k 0t | 1 -> k 1t | 2 -> k 2t], where [0t] is the
   tagged immediate, becomes [let tagged = Tag_immediate i in k tagged],
   or [k c] where [c] is known to be the tagged [i].
   [merge_arms] here is that function. Its inputs are the arms as they
   are after the first two slices, what is known of the primitives
   that have been computed ([computed], ../../rocq/FlambdaCseSpec.v),
   and the names for the variables that it may bind.

   What is stated.
   - Of the machine ([merge_arms_refines_statement]): what
     [merge_arms] answers refines the [Switch], in every state in
     which what is known holds ([all_computed],
     ../../rocq/FlambdaCseSpec.v). So does what
     [rebuild_switch] answers, which is [Invalid] for no arm, the
     merged term where there is one, and otherwise what the first
     slice makes of the arms ([rebuild_switch_refines_statement]); with
     nothing known that is a refinement of expressions
     ([rebuild_switch_nothing_known_refines_statement]). These are
     refinements and not equivalences: the [Switch] is undefined where
     the scrutinee is not the number of an arm, and the merged term is
     not.
   - Under a typing environment, after the first two slices
     ([simplify_switch_merging_refines_statement], and
     [simplify_switch_merging_of_meet_to_prove_refines_statement] at
     the transcribed meet that the contract is proved of): what
     [simplify_switch_merging] answers refines the [Switch], in the
     states whose static environment is valid and in which what is
     known holds ([expr_refines_under_valid_where]; with nothing known
     that is the notion of the first two slices,
     [expr_refines_under_valid_where_nothing_known_statement]). It is
     the function of the second slice with [rebuild_switch] where that
     has [switch_of_arms].
   - The functions commute with a renaming of variables
     ([merge_arms_rename_statement], [rebuild_switch_rename_statement],
     [simplify_switch_merging_rename_statement]): what is known and
     the names that are given are renamed with the term. The first
     two have no premise; the third has those of the second slice, on
     the meet and the rewrite, or on the join where the meet is the
     transcribed one that the contract is proved of
     ([simplify_switch_merging_of_meet_to_prove_rename_statement]).
   - Two terms from OCaml sources ([example_switch_index_statement],
     [example_switch_negate_statement]).

   What is assumed.
   - That what is known holds, in the state: each primitive that is
     known is one whose value stays, and the machine, run on it and
     its arguments in the state, gives the value that the variable
     holds and leaves the heap and the read record as they are
     ([all_computed], ../../rocq/FlambdaCseSpec.v, where it is what
     it means for what the pass that shares primitives knows to
     hold). It is a hypothesis about the caller, as the soundness of
     the meet is in the first slice. Nothing here says that what
     Simplify knows has it. Under a typing environment it is asked
     state by state, in the notion of refinement
     ([expr_refines_under_valid_where]), and not of every state that
     satisfies the environment: what is known may be of a primitive
     that reads a cell, and a typing environment says nothing of the
     heap. With nothing known it holds, and
     then nothing from before the [Switch] is used: a primitive is
     bound the first time an argument asks for it, and one that is
     bound for an argument may still be found for the next ones, where
     one of its arguments is a variable.
   - Under a typing environment, what the second slice assumes, and
     its header says why.
   Nothing is assumed of the names that are given: [merge_arms] tests
   that they are distinct and occur neither in the scrutinee, nor in
   the arms, nor in what is known ([names_new]), and merges nothing
   where they are not. Nor of the numbers of the arms, which it tests
   to be immediates ([in_range]): the machine does not ask it of the
   scrutinee of a [Switch], and [Tag_immediate] has no value on a
   number that is not one. The numbers of the compiler are immediates
   by their type.

   Where the model differs from the compiler, or says less.
   - The same argument in every arm. Simplify keeps, for each argument
     of each arm, the set of its aliases in the environment of the arm,
     and looks for a [Simple] that is in all the sets, among the names
     that the term still needs ([find_all_aliases],
     [recognize_invariant_argument], [filter_and_choose_alias]). Here
     the argument is invariant where it is the same [Simple] in every
     arm, so fewer arguments are. Where Simplify finds a variable that
     is equal to the number of each arm in the environment of that arm,
     the model may find the scrutinee or bind a primitive instead.
   - Poison. For Simplify an argument that is poison in an arm agrees
     with any alias, and is skipped when a slope and an offset are
     looked for. Here a poison constant is a constant like another: it
     is the same as itself only, and it is not affine. To say more
     needs a refinement in which poison is exempt.
   - Lookup tables. Where the arguments are constants or symbols of
     one kind, the numbers of the arms are consecutive, and there are
     three arms or more, Simplify makes a static array of the
     arguments and loads from it ([Lookup_table_argument]). This
     slice has no table, and nothing is merged there. The rewrite of
     the validation that merges into a load from a table that is in
     scope is FlambdaMergeTablesSpec.merge_tables.
   - The names. Simplify makes a new variable for each primitive that
     it binds. Here the names are given, and are taken in order; where
     there are not enough, nothing is merged. An argument binds three
     primitives at most.
   - What is known. For Simplify it is its environment of common
     subexpressions before the [Switch], of which it takes an alias
     that the term still needs; here it is a list of variables with
     the primitive that each holds, as in ../../rocq/FlambdaCseSpec.v,
     which stands for what that search answers. What is bound for an
     argument is added to it for the next ones, as Simplify adds it
     to its local environment. A primitive none of whose arguments is
     a variable is neither looked for nor added
     ([Flambda_primitive.Eligible_for_cse.create]).
   - The arms. In the compiler they are a map from numbers, which is
     read in the order of the numbers; here they are a list, which is
     read in its order. The statements hold of any list. On a list
     that is in another order the slope and the offset may be read off
     other arms; on a list with two arms for one number, which a map
     has not, no slope is read off those two.
   - Arms with arguments in different numbers, which the arms of one
     continuation have not in the compiler, are not merged.

   Not here: the arms that are removed on the way up ([rebuild_arm]:
   an arm whose continuation cannot be reached, a [goto] of a
   continuation that is itself an [Apply_cont]); the arms that are the
   same [Apply_cont] ([create_switch]); the counts that Simplify keeps
   of what it has removed and added.

   The proofs are in FlambdaSimplifySwitchMergeProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaClosuresSpec FlambdaNormaliseSpec FlambdaNominalSpec FlambdaCseSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeRenameSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaEnvRenameSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetTermsSpec FlambdaMeetRenameSpec FlambdaSimplifySwitchSpec
  FlambdaSimplifySwitchArmsSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyPassSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The arithmetic of immediates                                             *)
(* ======================================================================== *)

(* [Target_ocaml_int], on a 64-bit target: the integers of 63 bits.
   Addition, subtraction, multiplication and the opposite wrap.
   Division truncates towards zero, and wraps where the least integer
   is divided by minus one; [mod_] is the remainder of that division
   ([Z.rem]). *)
Definition immediate_width : Z := int_width Standard_naked_immediate.

Definition immediate_add (a b : Z) : Z := wrap immediate_width (a + b).

Definition immediate_sub (a b : Z) : Z := wrap immediate_width (a - b).

Definition immediate_mul (a b : Z) : Z := wrap immediate_width (a * b).

Definition immediate_neg (a : Z) : Z := wrap immediate_width (- a).

Definition immediate_div (a b : Z) : Z := wrap immediate_width (Z.quot a b).

(* ======================================================================== *)
(* An argument that is affine in the number of the arm                      *)
(* ======================================================================== *)

(* [affine_immediate_kind]. *)
Inductive affine_immediate_kind : Type :=
  | Affine_tagged
  | Affine_naked.

Definition affine_immediate_kind_eqb (kind kind' : affine_immediate_kind)
    : bool :=
  match kind, kind' with
  | Affine_tagged, Affine_tagged | Affine_naked, Affine_naked => true
  | _, _ => false
  end.

Definition affine_immediate_kind_to_standard_int
    (kind : affine_immediate_kind) : standard_int :=
  match kind with
  | Affine_tagged => Standard_tagged_immediate
  | Affine_naked => Standard_naked_immediate
  end.

(* [affine_argument_of_mergeable_arm]: what has been found of one
   argument in the arms read so far. [Affine_constant] is [Constant] of
   the compiler: one arm has been read, with that number and that
   argument. *)
Inductive affine_argument_of_mergeable_arm : Type :=
  | Not_affine
  | Maybe_affine
  | Affine_constant (kind : affine_immediate_kind) (input output : Z)
  | Affine (kind : affine_immediate_kind) (offset slope : Z).

(* [recognize_affine_immediate_const]. A constant that is not in the
   range of immediates is not one of the compiler's, and is not taken. *)
Definition recognize_affine_immediate_const (c : const)
    : option (affine_immediate_kind * Z) :=
  match c with
  | Naked_immediate i =>
    if in_range immediate_width i then Some (Affine_naked, i) else None
  | Tagged_immediate i =>
    if in_range immediate_width i then Some (Affine_tagged, i) else None
  | _ => None
  end.

(* [recognize_affine_argument]: one more arm, of number [discr] and
   whose argument is the constant. The slope and the offset come from
   the first two arms. Where the kinds of two arms differ the compiler
   stops with an error; here the argument is not affine. Two arms for
   one number, which the compiler has not, are not affine either. *)
Definition recognize_affine_argument
    (affine : affine_argument_of_mergeable_arm) (discr : Z) (c : const)
    : affine_argument_of_mergeable_arm :=
  match affine, recognize_affine_immediate_const c with
  | Not_affine, _ | _, None => Not_affine
  | Maybe_affine, Some (kind, imm) => Affine_constant kind discr imm
  | Affine_constant kind arm arg, Some (imm_kind, imm) =>
    let arg_diff := immediate_sub imm arg in
    let arm_diff := immediate_sub discr arm in
    if affine_immediate_kind_eqb kind imm_kind && negb (arm_diff =? 0)
       && (Z.rem arg_diff arm_diff =? 0)
    then
      let slope := immediate_div arg_diff arm_diff in
      let offset := immediate_sub arg (immediate_mul slope arm) in
      Affine kind offset slope
    else Not_affine
  | Affine kind offset slope, Some (imm_kind, imm) =>
    if affine_immediate_kind_eqb imm_kind kind
       && (imm =? immediate_add (immediate_mul discr slope) offset)
    then Affine kind offset slope
    else Not_affine
  end.

(* ======================================================================== *)
(* The arguments that can be merged                                         *)
(* ======================================================================== *)

(* [mergeable_argument], without [Lookup_table_argument]. *)
Inductive mergeable_argument : Type :=
  | Invariant_argument (s : simple)
  | Tagged_scrutinee
  | Not_scrutinee
  | Affine_argument (kind : affine_immediate_kind) (offset slope : Z).

(* [enough_arms]. *)
Definition enough_arms (num_arms : nat) (argument : mergeable_argument)
    : bool :=
  match argument with
  | Invariant_argument _ | Tagged_scrutinee | Not_scrutinee => true
  | Affine_argument _ _ _ => (3 <=? num_arms)%nat
  end.

Definition same_simple (s s' : simple) : bool :=
  if simple_eq_dec s s' then true else false.

(* One argument, in each arm: the number of the arm and the argument
   there, in the order of the arms. *)
Definition argument_of_arms : Type := list (Z * simple).

(* [recognize_invariant_argument], for an argument that is the same
   [Simple] in every arm (the header says what the compiler does). *)
(* CR mshinwell: this is less than the compiler's test, which is on
   aliases: an argument is invariant for Simplify where one [Simple]
   is an alias of it in the environment of each arm, and poison is an
   alias of anything. So the model merges fewer [Switch]es than the
   compiler, and where both merge the terms may differ (the compiler
   passes a variable that it knows to be equal to the argument in each
   arm; the model passes the scrutinee or binds a primitive). What is
   stated is of the model's function, and is a refinement either way.
   To follow the compiler, the second slice has to answer the
   environment of each arm and this test has to read the aliases
   there; and for poison the statement has to be a refinement in
   which poison is exempt (decision D26), which there is not yet. It
   would be wrong to take this file as saying that the compiler's
   test on aliases is sound: it does not transcribe it. *)
Definition recognize_invariant_argument (arg : argument_of_arms)
    : option simple :=
  match arg with
  | [] => None
  | (_, s) :: rest =>
    if forallb (fun entry => same_simple s (snd entry)) rest
    then Some s
    else None
  end.

(* [check_arm] of [recognize_mergeable_argument], as far as the affine
   argument. An argument that is a symbol is not affine. Of one that
   is a variable the compiler stops looking; here it is not affine,
   which comes to the same as there are no lookup tables. *)
Definition check_arm (affine : affine_argument_of_mergeable_arm)
    (entry : Z * simple) : affine_argument_of_mergeable_arm :=
  match snd entry with
  | Const c => recognize_affine_argument affine (fst entry) c
  | Var _ | Symbol _ => Not_affine
  end.

(* [recognize_mergeable_argument]. The identity on naked immediates is
   the scrutinee; on tagged immediates it is the scrutinee tagged. The
   negation is the affine function of offset 1 and slope -1 on tagged
   immediates, where the numbers of the arms are among 0 and 1
   ([affine_is_boolean_not]). *)
Definition recognize_mergeable_argument (scrutinee : simple)
    (arg : argument_of_arms) : option mergeable_argument :=
  match recognize_invariant_argument arg with
  | Some s => Some (Invariant_argument s)
  | None =>
    match fold_left check_arm Maybe_affine arg with
    | Affine kind offset slope =>
      if (offset =? 0) && (slope =? 1)
      then
        match kind with
        | Affine_naked => Some (Invariant_argument scrutinee)
        | Affine_tagged => Some Tagged_scrutinee
        end
      else
        match kind with
        | Affine_tagged =>
          if (offset =? 1) && (slope =? -1)
             && forallb (fun entry => (fst entry =? 0) || (fst entry =? 1))
                  arg
          then Some Not_scrutinee
          else Some (Affine_argument kind offset slope)
        | Affine_naked => Some (Affine_argument kind offset slope)
        end
    | Not_affine | Maybe_affine | Affine_constant _ _ _ => None
    end
  end.

(* The first argument of each arm, if every arm has one; and the arms
   without their first arguments. *)
Fixpoint first_arguments (rows : list (Z * list simple))
    : option argument_of_arms :=
  match rows with
  | [] => Some []
  | (_, []) :: _ => None
  | (i, arg :: _) :: rows =>
    let* args := first_arguments rows in
    Some ((i, arg) :: args)
  end.

Definition other_arguments (rows : list (Z * list simple))
    : list (Z * list simple) :=
  map (fun row => (fst row, tl (snd row))) rows.

(* Every argument can be merged, and there are enough arms for each
   (the test of [rebuild_switch] on [mergeable_args]). [rows] has, for
   each arm, its number and its arguments. *)
Fixpoint recognize_mergeable_arguments (scrutinee : simple) (num_arms : nat)
    (num_args : nat) (rows : list (Z * list simple))
    : option (list mergeable_argument) :=
  match num_args with
  | O => Some []
  | S num_args =>
    let* arg := first_arguments rows in
    let* argument := recognize_mergeable_argument scrutinee arg in
    if enough_arms num_arms argument
    then
      let* arguments :=
        recognize_mergeable_arguments scrutinee num_arms num_args
          (other_arguments rows)
      in
      Some (argument :: arguments)
    else None
  end.

(* ======================================================================== *)
(* Binding the primitives                                                   *)
(* ======================================================================== *)

(* What the rebuilding carries from one argument to the next: the
   [Let]s that it has made, the last one first; what is known of the
   primitives that have been computed, with those; and the names that
   are left. *)
Record merging : Type := Merging {
  merging_lets : list (variable * flambda_kind * prim * list simple);
  merging_known : computed;
  merging_fresh : list variable
}.

Definition simple_is_var (s : simple) : bool :=
  match s with
  | Var _ => true
  | Symbol _ | Const _ => false
  end.

(* [let$] on a primitive: a variable that is known to hold it, or a
   [Let] of it to the next name, which is then known to hold it.
   [None] where no name is left. *)
Definition bind_prim (m : merging) (kind : flambda_kind) (p : prim)
    (args : list simple) : option (simple * merging) :=
  let eligible := existsb simple_is_var args in
  match
    if eligible then computed_by (merging_known m) p args else None
  with
  | Some x => Some (Var x, m)
  | None =>
    match merging_fresh m with
    | [] => None
    | x :: fresh =>
      Some
        (Var x,
         Merging ((x, kind, p, args) :: merging_lets m)
           (if eligible
            then (x, (p, args)) :: merging_known m
            else merging_known m)
           fresh)
    end
  end.

(* [bound_tagged_scrutinee]. *)
Definition bind_tagged_scrutinee (scrutinee : simple) (m : merging)
    : option (simple * merging) :=
  bind_prim m Value (Unary Tag_immediate) [scrutinee].

(* [rebuild_affine_expr]: the scrutinee, at the kind of the argument,
   times the slope, added to the offset. A negative slope is negated
   and subtracted. The multiplication by 1 and the addition of 0 are
   not made. *)
Definition rebuild_affine_expr (kind : affine_immediate_kind)
    (offset slope : Z) (scrutinee : simple) (m : merging)
    : option (simple * merging) :=
  let standard_int := affine_immediate_kind_to_standard_int kind in
  let let_kind := kind_of_standard_int standard_int in
  let must_negate := slope <? 0 in
  let slope := if must_negate then immediate_neg slope else slope in
  let* '(scaled_arg, m) :=
    if slope =? 1
    then Some (scrutinee, m)
    else
      bind_prim m let_kind (Binary (Int_arith standard_int Mul))
        [scrutinee; Const (const_of_int standard_int slope)]
  in
  if (offset =? 0) && negb must_negate
  then Some (scaled_arg, m)
  else
    bind_prim m let_kind
      (Binary (Int_arith standard_int (if must_negate then Sub else Add)))
      [Const (const_of_int standard_int offset); scaled_arg].

(* [rebuild_mergeable_argument]: the [Simple] that the argument
   becomes. *)
Definition rebuild_mergeable_argument (scrutinee : simple) (m : merging)
    (argument : mergeable_argument) : option (simple * merging) :=
  match argument with
  | Invariant_argument s => Some (s, m)
  | Affine_argument Affine_naked offset slope =>
    rebuild_affine_expr Affine_naked offset slope scrutinee m
  | Affine_argument Affine_tagged offset slope =>
    let* '(tagged_scrutinee, m) := bind_tagged_scrutinee scrutinee m in
    rebuild_affine_expr Affine_tagged offset slope tagged_scrutinee m
  | Tagged_scrutinee => bind_tagged_scrutinee scrutinee m
  | Not_scrutinee =>
    let* '(tagged_scrutinee, m) := bind_tagged_scrutinee scrutinee m in
    bind_prim m Value (Unary Boolean_not) [tagged_scrutinee]
  end.

(* [rebuild_merged_switch]: the arguments one after the other. *)
Fixpoint rebuild_mergeable_arguments (scrutinee : simple) (m : merging)
    (arguments : list mergeable_argument) : option (list simple * merging) :=
  match arguments with
  | [] => Some ([], m)
  | argument :: arguments =>
    let* '(arg, m) := rebuild_mergeable_argument scrutinee m argument in
    let* '(args, m) := rebuild_mergeable_arguments scrutinee m arguments in
    Some (arg :: args, m)
  end.

(* An expression under [Let]s, the last one innermost. *)
Fixpoint bind_lets (lets : list (variable * flambda_kind * prim * list simple))
    (body : expr) : expr :=
  match lets with
  | [] => body
  | (x, kind, p, args) :: lets =>
    bind_lets lets (Let x kind (Prim p args) body)
  end.

(* ======================================================================== *)
(* Merging the arms                                                         *)
(* ======================================================================== *)

(* [Mergeable] of [rebuild_arm]: no arm has a trap action, and every
   arm applies the continuation of the first, to as many arguments. The
   continuation and the number of arguments. *)
Definition mergeable_arms (arms : list (Z * apply_cont))
    : option (continuation * nat) :=
  match arms with
  | [] => None
  | (_, ac) :: _ =>
    if forallb
         (fun arm =>
            match ac_trap_action (snd arm) with
            | None => true
            | Some _ => false
            end
            && Pos.eqb (ac_continuation (snd arm)) (ac_continuation ac)
            && Nat.eqb (length (ac_args (snd arm))) (length (ac_args ac)))
         arms
    then Some (ac_continuation ac, length (ac_args ac))
    else None
  end.

(* A name is new: it occurs neither in the scrutinee, nor in an
   argument of an arm, nor in what is known. *)
Definition name_new (known : computed) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (x : variable) : bool :=
  negb (var_in_simple x scrutinee)
  && forallb (fun arm => negb (var_in_simples x (ac_args (snd arm)))) arms
  && forallb
       (fun entry =>
          negb (Pos.eqb x (fst entry))
          && negb (var_in_simples x (snd (snd entry))))
       known.

Definition names_new (fresh : list variable) (known : computed)
    (scrutinee : simple) (arms : list (Z * apply_cont)) : bool :=
  distinct fresh && forallb (name_new known scrutinee arms) fresh.

(* The merged [Switch] ([switch_merged] and [rebuild_merged_switch] of
   [rebuild_switch]). [None] where the arms are not merged. *)
Definition merge_arms (known : computed) (fresh : list variable)
    (scrutinee : simple) (arms : list (Z * apply_cont)) : option expr :=
  let* '(cont, num_args) := mergeable_arms arms in
  if forallb (fun arm => in_range immediate_width (fst arm)) arms
     && names_new fresh known scrutinee arms
  then
    let* arguments :=
      recognize_mergeable_arguments scrutinee (length arms) num_args
        (map (fun arm => (fst arm, ac_args (snd arm))) arms)
    in
    let* '(args, m) :=
      rebuild_mergeable_arguments scrutinee (Merging [] known fresh)
        arguments
    in
    Some
      (bind_lets (merging_lets m)
         (Apply_cont (Apply_cont_expr cont args None)))
  else None.

(* [rebuild_switch], of the arms that are left: [Invalid] for none,
   the merged term where the arms are merged, and otherwise what the
   first slice makes of them. *)
Definition rebuild_switch (known : computed) (fresh : list variable)
    (scrutinee : simple) (arms : list (Z * apply_cont)) : expr :=
  match arms with
  | [] => Invalid
  | _ :: _ =>
    match merge_arms known fresh scrutinee arms with
    | Some e => e
    | None => switch_of_arms scrutinee arms
    end
  end.

(* ======================================================================== *)
(* Of the machine                                                           *)
(* ======================================================================== *)

(* That what is known holds in a state is [all_computed] of
   ../../rocq/FlambdaCseSpec.v: each primitive is one whose value
   stays, and run by the machine in the state on its arguments it
   gives the value that its variable holds, and leaves the heap and
   the read record as they are. *)

(* The merged term refines the [Switch]. *)
Definition merge_arms_refines_statement : Prop :=
  ∀ (known : computed) (fresh : list variable) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    merge_arms known fresh scrutinee arms = Some e ->
    all_computed known vars statics hp rd ->
    refines (State (Switch scrutinee arms) vars conts statics hp rd)
      (State e vars conts statics hp rd).

Definition rebuild_switch_refines_statement : Prop :=
  ∀ (known : computed) (fresh : list variable) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    all_computed known vars statics hp rd ->
    refines (State (Switch scrutinee arms) vars conts statics hp rd)
      (State (rebuild_switch known fresh scrutinee arms) vars conts statics hp
         rd).

(* With nothing known, in every state. *)
Definition rebuild_switch_nothing_known_refines_statement : Prop :=
  ∀ (fresh : list variable) (scrutinee : simple)
    (arms : list (Z * apply_cont)),
    expr_refines (Switch scrutinee arms)
      (rebuild_switch [] fresh scrutinee arms).

(* ======================================================================== *)
(* After the first two slices                                               *)
(* ======================================================================== *)

(* Refinement in the states that satisfy a typing environment, whose
   static environment is valid, and in which what is known holds. *)
Definition expr_refines_under_valid_where (tenv : Typing_env.t)
    (known : computed) (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    statics_valid statics ->
    env_satisfies_strictly tenv statics vars ->
    all_computed known vars statics hp rd ->
    refines (State source vars conts statics hp rd)
      (State target vars conts statics hp rd).

(* With nothing known it is [expr_refines_under_valid]. *)
Definition expr_refines_under_valid_where_nothing_known_statement : Prop :=
  ∀ (tenv : Typing_env.t) (source target : expr),
    expr_refines_under_valid_where tenv [] source target
    <-> expr_refines_under_valid tenv source target.

(* [simplify_switch] with [rebuild_switch]: the arms are pruned and
   simplified as in the second slice ([simplify_arms]), and what is
   left is merged where it can be. *)
Definition simplify_switch_merging (meet : meet_function)
    (rewrite : apply_cont_rewrite) (known : computed) (fresh : list variable)
    (tenv : Typing_env.t) (scrutinee : simple) (arms : list (Z * apply_cont))
    : option expr :=
  let* answer :=
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
  in
  let* '(scrutinee_ty, canonical) := answer in
  let* arms := simplify_arms meet rewrite tenv scrutinee_ty arms in
  Some (rebuild_switch known fresh canonical arms).

Definition simplify_switch_merging_refines_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite) (known : computed)
    (fresh : list variable) (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (e : expr),
    meet_sound meet ->
    meet_keeps_terms meet ->
    apply_cont_rewrite_sound rewrite ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    simplify_switch_merging meet rewrite known fresh tenv scrutinee arms
    = Some e ->
    expr_refines_under_valid_where tenv known (Switch scrutinee arms) e.

(* At the transcribed meet that the contract is proved of. *)
Definition simplify_switch_merging_of_meet_to_prove_refines_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (rewrite : apply_cont_rewrite)
    (known : computed) (fresh : list variable) (tenv : Typing_env.t)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (e : expr),
    join_scoped_envs_sound join ->
    apply_cont_rewrite_sound rewrite ->
    typing_env_fits tenv ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv scrutinee = true ->
    simplify_switch_merging (meet_to_prove join fuel) rewrite known fresh tenv
      scrutinee arms
    = Some e ->
    expr_refines_under_valid_where tenv known (Switch scrutinee arms) e.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* What is known and the names that are given are renamed with the
   term ([rename_computed], ../../rocq/FlambdaCseSpec.v). *)
Definition merge_arms_rename_statement : Prop :=
  ∀ (r : renaming) (known : computed) (fresh : list variable)
    (scrutinee : simple) (arms : list (Z * apply_cont)),
    merge_arms (rename_computed r known) (map (rename_variable r) fresh)
      (rename_simple r scrutinee) (rename_arms r arms)
    = option_map (rename_expr r) (merge_arms known fresh scrutinee arms).

Definition rebuild_switch_rename_statement : Prop :=
  ∀ (r : renaming) (known : computed) (fresh : list variable)
    (scrutinee : simple) (arms : list (Z * apply_cont)),
    rebuild_switch (rename_computed r known) (map (rename_variable r) fresh)
      (rename_simple r scrutinee) (rename_arms r arms)
    = rename_expr r (rebuild_switch known fresh scrutinee arms).

(* For a meet and a rewrite that commute with the renaming, as in the
   second slice. *)
Definition simplify_switch_merging_rename_statement : Prop :=
  ∀ (meet : meet_function) (rewrite : apply_cont_rewrite) (r : renaming)
    (known : computed) (fresh : list variable) (tenv : Typing_env.t)
    (scrutinee : simple) (arms : list (Z * apply_cont)),
    meet_function_equivariant meet ->
    apply_cont_rewrite_equivariant rewrite ->
    simplify_switch_merging meet rewrite (rename_computed r known)
      (map (rename_variable r) fresh) (rename_typing_env r tenv)
      (rename_simple r scrutinee) (rename_arms r arms)
    = option_map (rename_expr r)
        (simplify_switch_merging meet rewrite known fresh tenv scrutinee arms).

(* At the transcribed meet that the contract is proved of, with a join
   that commutes with renaming
   ([meet_to_prove_equivariant_statement], FlambdaMeetRenameSpec.v). *)
Definition simplify_switch_merging_of_meet_to_prove_rename_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat) (rewrite : apply_cont_rewrite)
    (r : renaming) (known : computed) (fresh : list variable)
    (tenv : Typing_env.t) (scrutinee : simple) (arms : list (Z * apply_cont)),
    join_scoped_envs_equivariant join ->
    apply_cont_rewrite_equivariant rewrite ->
    simplify_switch_merging (meet_to_prove join fuel) rewrite
      (rename_computed r known) (map (rename_variable r) fresh)
      (rename_typing_env r tenv) (rename_simple r scrutinee)
      (rename_arms r arms)
    = option_map (rename_expr r)
        (simplify_switch_merging (meet_to_prove join fuel) rewrite known fresh
           tenv scrutinee arms).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The two terms are what Simplify has of these OCaml sources when it
   comes back up to the [Switch], and what it makes of them: the
   sources were compiled and the terms before and after Simplify read
   ([-drawfexpr-to], [-dfexpr-to]). The variables are those of
   FlambdaTypingEnvExamplesSpec.v, and two more for the names. *)
Definition example_tagged : variable := 9%positive.

Definition example_negated : variable := 10%positive.

Definition example_tagged_arm (k : continuation) (i : Z) : apply_cont :=
  Apply_cont_expr k [Const (Tagged_immediate i)] None.

(* [type colour = Red | Green | Blue] and
   [let index : colour -> int = fun c ->
      match c with Red -> 0 | Green -> 1 | Blue -> 2].
   The body of [index] reaches Simplify as
   [let i = Untag_immediate x in
    switch i with 0 -> k 0t | 1 -> k 1t | 2 -> k 2t],
   [x] being the parameter. Each argument is the number of its arm,
   tagged, so the argument is the scrutinee tagged
   ([Tagged_scrutinee]). After the [Let], Simplify knows that [x] holds
   [Tag_immediate i], and binds nothing: the body becomes [k x]. *)
Definition example_switch_index (body : expr) : expr :=
  Let example_i (Naked_number Kind_naked_immediate)
    (Prim (Unary Untag_immediate) [Var example_x]) body.

Definition example_index_known : computed :=
  [(example_x, (Unary Tag_immediate, [Var example_i]))].

Definition example_index_arms (k : continuation) : list (Z * apply_cont) :=
  [(0, example_tagged_arm k 0); (1, example_tagged_arm k 1);
   (2, example_tagged_arm k 2)].

Definition example_switch_index_statement : Prop :=
  ∀ k : continuation,
    rebuild_switch example_index_known [example_tagged; example_negated]
      (Var example_i) (example_index_arms k)
    = Apply_cont (Apply_cont_expr k [Var example_x] None)
    ∧ expr_refines
        (example_switch_index (Switch (Var example_i) (example_index_arms k)))
        (example_switch_index
           (Apply_cont (Apply_cont_expr k [Var example_x] None))).

(* With [colour] as above,
   [let[@inline] code : colour -> int = fun c ->
      match c with Red -> 1 | Green -> 0 | Blue -> opaque 7] and
   [let f : int -> int = fun x -> code (if x < 7 then Green else Red)].
   Once [code] is inlined in [f], the [Switch] of [code] is on the
   comparison:
   [let i = (x < 7) in switch i with 0 -> k 1t | 1 -> k 0t | 2 -> k2],
   which is the term of the second example of the first slice with
   other arms, in the same environment and with the same meet
   ([example_meet], FlambdaSimplifySwitchSpec.v). The arm for 2 is
   dropped, as there. The argument of the two arms that are left is the
   negation of the scrutinee ([Not_scrutinee]). Nothing is known that
   the term still needs, and Simplify binds two primitives:
   [let i = (x < 7) in let tagged = Tag_immediate i in
    let negated = Boolean_not tagged in k negated]. *)
Definition example_negate_arms (k k2 : continuation)
    : list (Z * apply_cont) :=
  [(0, example_tagged_arm k 1); (1, example_tagged_arm k 0);
   (2, Apply_cont_expr k2 [] None)].

Definition example_negate_merged (k : continuation) : expr :=
  Let example_tagged Value (Prim (Unary Tag_immediate) [Var example_i])
    (Let example_negated Value
       (Prim (Unary Boolean_not) [Var example_tagged])
       (Apply_cont (Apply_cont_expr k [Var example_negated] None))).

Definition example_switch_negate_statement : Prop :=
  ∀ k k2 : continuation,
    expr_refines_under_valid
      (Typing_env.add_variable_definition Typing_env.create example_x Value
         Name_mode.normal)
      (example_switch_comparison (example_negate_arms k k2))
      (Let example_i (Naked_number Kind_naked_immediate)
         (Prim
            (Binary
               (Int_comp Standard_tagged_immediate
                  (Yielding_bool (Int_lt Signed))))
            [Var example_x; Const (Tagged_immediate 7)])
         (example_negate_merged k)).
