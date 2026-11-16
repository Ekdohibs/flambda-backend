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

(* A [Switch] whose arms all apply one continuation, to arguments
   that are the elements of static tables at the number of the arm: a
   normalisation that refines.

   This file is to be reviewed: [merge_tables] and what it is made of,
   and the statements. The rewrite is one of the normalisations of the
   term before Simplify that the validation of
   middle_end/flambda2/tests/semantics/ applies, once in a round for a
   whole term (FlambdaValidateSpec.normalise_params_round).

   What Simplify does. Where the arms of a [Switch] all apply the same
   continuation, and an argument is in every arm a constant or a
   symbol, is not the same in every arm and is not computed from the
   number of the arm in one of the ways of
   FlambdaMergeSwitchesSpec.v, Simplify may load it from a table
   ([Lookup_table_argument] of
   middle_end/flambda2/simplify/simplify_switch_expr.ml): the numbers
   of the arms are then [d], [d + 1], and so on with none missing,
   and there are three arms or more ([recognize_mergeable_argument],
   [enough_arms]). The table is a symbol bound to the immutable array
   of the arguments in the order of the arms, a constant that Simplify
   lifts, so that its binding is with the static bindings of the unit
   and not where the [Switch] was. The argument is then computed by
   ([rebuild_mergeable_argument])

     tagged = Tag_immediate scrutinee
     index  = tagged                                 where d = 0
            = Int_arith tagged_immediate Add (tagged, -d)   otherwise
     arg    = Array_load (kind, load kind, Immutable) (table, index)

   where, for an argument of kind value, the array kind and the load
   kind are those of immediates if every element is a tagged immediate
   constant, and those of values otherwise. The table is then an
   immutable array of values, and a poison constant of kind value in
   an arm is that constant in the table. For an argument that is in
   every arm a constant of one kind of naked number (naked immediate,
   float32, float, int8, int16, int32, int64, nativeint, vec128,
   vec256, vec512 or mask) the table is an immutable array of those
   numbers, with the array kind and the load kind of that array, and a
   poison constant in an arm is some number in the table
   ([create_lookup_table_array_const]).

   What of that is here. The model has no static array but that of
   values (FlambdaModel.Static_const_immutable_value_array), so only
   the tables for an argument of kind value are tables here. An
   argument that is in some arm a constant that is not of kind value,
   a poison of such a kind too, is not loaded from a table: an array
   that has that constant is not one of immediates, and the load of
   values does not take an element that is not of kind value
   ([all_immediates], [element_taken]). Such an argument may still be
   computed from the number of the arm in one of the ways of
   FlambdaMergeSwitchesSpec.v, beside an argument that is loaded (see
   "A way for each argument" below). A poison constant of kind value
   is an element as another is: it is not a tagged immediate, the
   load of values takes it, and an arm whose argument is that
   constant is loaded from a table that has it at the index of the
   arm.

   What is here. [merge_tables] merges a [Switch] where tables that
   are in scope have the arguments of its arms that are loaded: one
   table every argument that differs from arm to arm or, argument by
   argument, a table each one that is not the same in every arm and
   is not computed from the number of the arm. Nothing is guessed
   here: the tables are read off the term. The term before Simplify
   has not the tables that Simplify makes. The validation binds, at
   the head of the term before, the tables that a lifting is given,
   where the checks of FlambdaUnusedHeadStaticsSpec.add_head_tables
   pass (FlambdaValidateSpec.with_lifting; where they fail the term
   is left as it is), and normalises the term after that. So in the
   comparisons of a term with a lifting a [Switch] is merged where
   tables that were bound so, or arrays that the term binds itself,
   have those arguments of its arms. In the other comparisons it is
   merged only where the term binds such arrays itself.

   The tables in scope ([known_tables], [tables_under_statics]). A
   table is a symbol that a [Let_static] binds to an immutable array
   of values whose elements are constants and symbols, with those
   elements. The rewrite carries the tables of the groups of static
   bindings that it has gone under; the body of code is rewritten
   with the tables of the groups that its binding is under and of its
   own group, as in FlambdaUnboxInScopeSpec.rewrite_code_in_scope.

   How a [Switch] is merged where one table has its arms
   ([merge_switch_with_tables], [merge_switch_by_table]). Only a
   [Switch] that FlambdaMergeSwitchesSpec.merge_switch leaves is
   merged: an argument that is computed from the number of the arm
   is not loaded from a table by Simplify either. The scrutinee must
   be a variable and there must be three arms or more. The
   application that the [Switch] may become applies the continuation
   of the first arm with no trap action; its arguments are those of
   the first arm where every arm has the same, and the variable of
   the scrutinee where one has not ([merged_across]; the elements of
   a table may be the same in the first two arms, so it is not
   enough to look at those as [merge_switch] does), and the variable
   must be among them. The first table in scope is taken that has as
   many elements as there are arms and of which every arm is that
   application with an element of the table in place of the variable
   ([arm_is_loaded]): the element at the index that the primitives
   compute of the number of the arm ([table_index],
   [table_element]), [d] being the number of the first arm, and that
   element must be one that the load kind takes ([element_taken]).
   The [Switch] becomes the application under the [Let]s of the
   three primitives, the second of which is not made where [d] is 0
   ([bind_table_load]). Each arm is tested on its own, so the
   refinement does not depend on the numbers of the arms being in
   order, on the length of the table or on how the merged arguments
   and the offset were found.

   The variable of the scrutinee is bound again by each [Let], as in
   [merge_switch], so nothing is named, and the one table serves
   every argument that differs from arm to arm.

   A way for each argument ([merge_switch_by_table_ways]). A [Switch]
   two of whose arguments are loaded from two tables, or one of whose
   arguments is loaded from a table and another computed from the
   number of the arm, has no one table for its arms, and Simplify
   merges it, deciding for each argument. So where no table has the
   arms a way is looked for for each argument
   ([table_way_of_column]): one of the ways of
   FlambdaMergeSwitchesSpec.way_of_column where there is one, as
   Simplify looks for those first; else the first table in scope
   that has as many elements as there are arms and has the argument
   of each arm at the index of that arm, an element that the load
   kind takes ([table_has_column]). The scrutinee must be a variable
   and there must be three arms or more, as for one table. One
   argument at least must be loaded: a [Switch] none of whose
   arguments is, is left to FlambdaMergeSwitchesSpec.merge_switches.
   Each argument that is computed or loaded is bound to a name by
   the [Let]s of its way, those of a table being the three above
   with the name in place of the variable of the scrutinee, which is
   not bound again ([bind_table_ways], [bind_table_load_to]). The
   names are the parameters of the continuation that the first arm
   applies, as in FlambdaMergeSwitchesSpec.merge_switch_by_arguments,
   so nothing is named here either. That continuation must be bound
   around the [Switch], in the term or in the body of code that the
   [Switch] is in: the return continuation of code is not one. The
   names must be usable ([table_names_usable]): one for each
   argument, no two the same, none the scrutinee, and none in an
   argument that is the same in every arm. Every arm must then apply
   that continuation, with no trap action, to the arguments that the
   ways give for its number ([arm_has_table_ways]): each arm is
   tested on its own here too.

   What it costs. With no table in scope a [Switch] is left as it
   is, and nothing is computed of it ([merge_switch_with_tables]);
   the term is still gone through. With a table in scope the tables
   are looked at before [merge_switch] is computed: of a [Switch] on
   a variable that has three arms or more, the merged arguments are
   computed, whether the variable is among them and, where it is,
   the offset; then the tables are tested in turn until one has the
   arms, and [merge_switch] is computed only where one has. Where
   none has, and the continuation of the first arm is bound around,
   the way of each argument is looked for in turn, the tables being
   tested only for an argument that has no way without one, and that
   stops at the first argument that has no way at all. The offset,
   and whether the elements of a table are all immediates, are not
   computed again for each arm: the offset is computed once for a
   [Switch], and whether the elements are all immediates where a
   table is tested and again where it is chosen.

   Where it goes ([merge_tables_under], [merge_tables_knowing],
   [merge_tables_in_code]). The first goes down a term, into the
   handlers of continuations, recursive ones too, and not into code,
   with the continuations that are bound around and their parameters
   (FlambdaMergeSwitchesSpec.known_continuations): a continuation is
   known in the body of its [Let_cont], and a recursive one in the
   handlers of its group too. The second is the first with no
   continuation known at the start; the third applies the second to
   the body of every piece of code, with the tables in scope there.
   [merge_tables] is the second and the third, with no table known at
   the start.

   The result refines the term under all environments
   ([merge_tables_refines_statement]): a group of static bindings
   that has a value binds each of its symbols to its constant, a name
   that is bound already being undefined, and no binding is ever
   removed, so a table that is in scope is bound to its elements
   wherever the rewrite reads it, in the body of code too, which runs
   after its group is bound. The two parts are stated where the
   tables are bound ([tables_bound];
   [merge_tables_knowing_refines_statement],
   [merge_tables_in_code_refines_statement]), and the first of them
   whatever continuations are taken as known, with whatever names
   ([merge_tables_under_refines_statement]). It is not an
   equivalence, as for [merge_switches]: the [Switch] is undefined
   where the scrutinee is the number of no arm.

   It commutes with renaming ([merge_tables_equivariant_statement]):
   it makes no name, a renaming moves no symbol, and the elements of
   a table have no variable. A term that binds no symbol to an
   immutable array is given back as it is
   ([merge_tables_without_tables_statement]). That it keeps scope is
   stated in FlambdaMergeTablesScopeSpec.v.

   Four terms as checks of the definitions
   ([merge_tables_example_statement];
   [merge_tables_equal_elements_example_statement], a table whose first
   two elements are the same;
   [merge_tables_two_tables_example_statement], two arguments loaded
   from two tables; [merge_tables_table_and_line_example_statement],
   an argument computed beside one that is loaded).

   The proofs are in FlambdaMergeTablesProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNormaliseSpec
  FlambdaMergeSwitchesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The tables in scope                                                      *)
(* ======================================================================== *)

(* The symbols in scope that are bound to an immutable array of values
   whose elements are constants and symbols, each with its elements. *)
Definition known_tables : Type := list (symbol * list simple).

(* An element that has a value whatever the variables hold: a constant
   or a symbol. *)
Definition element_closed (element : simple) : bool :=
  is_some (eval_simple [] element).

Definition table_of_binding (binding : static_binding) : known_tables :=
  match binding with
  | Bind_symbol s (Static_const_immutable_value_array elements) =>
    if forallb element_closed elements then [(s, elements)] else []
  | Bind_symbol _ _ | Bind_code _ _ | Bind_deleted_code _ => []
  end.

(* The tables in scope under a group of static bindings: those of the
   group, then those that were in scope. *)
Definition tables_under_statics (tables : known_tables)
    (bindings : list static_binding) : known_tables :=
  flat_map table_of_binding bindings ++ tables.

(* ======================================================================== *)
(* The element of a table that an arm gives                                 *)
(* ======================================================================== *)

(* What is added to the tagged scrutinee where the first arm is not
   that of 0: the opposite of the number of the first arm, which
   wraps as that of Simplify does. It is computed once for a [Switch]
   and given to what follows as [offset]; nothing below depends on its
   being this number, as every arm is tested with it. *)
Definition table_offset (first : Z) : Z :=
  wrap (int_width Standard_tagged_immediate) (- first).

(* The index that the primitives compute of the number of an arm:
   [Tag_immediate] has a value on an immediate only, and the addition
   wraps, and has none where the constant is not a tagged immediate. *)
Definition table_index (first offset i : Z) : option Z :=
  let width := int_width Standard_tagged_immediate in
  if in_range (int_width Standard_naked_immediate) i
  then
    if first =? 0 then Some i
    else if in_range width offset
    then Some (wrap width (i + offset))
    else None
  else None.

(* The element that the load gives for the arm of number [i]: there is
   none where the index is not one of the table. *)
Definition table_element (first offset : Z) (elements : list simple) (i : Z)
    : option simple :=
  let* index := table_index first offset i in
  if (0 <=? index) && (index <? Z.of_nat (length elements))
  then nth_error elements (Z.to_nat index)
  else None.

(* The kinds of the load: those of immediates where every element is a
   tagged immediate, those of values otherwise. *)
Definition all_immediates (elements : list simple) : bool :=
  forallb
    (fun element =>
       match eval_simple [] element with
       | Some w => is_tagged_immediate w
       | None => false
       end)
    elements.

Definition table_array_kind (immediates : bool) : array_kind :=
  if immediates then Array_immediates else Array_values.

Definition table_load_kind (immediates : bool) : array_load_kind :=
  if immediates then Load_immediates else Load_values.

(* The load has a value on the element: it is one that the load kind
   takes. Where every element is a tagged immediate, which is where
   [immediates] is to be [true], each is taken by the load of
   immediates; otherwise the element must be of kind [Value]. *)
Definition element_taken (immediates : bool) (element : simple) : bool :=
  immediates
  || match eval_simple [] element with
     | Some w => array_element_ok Array_values w
     | None => false
     end.

(* ======================================================================== *)
(* The application that the arms are merged into                            *)
(* ======================================================================== *)

(* An element in place of a variable. *)
Definition element_for (v : variable) (element : simple) (s : simple)
    : simple :=
  match s with
  | Var x => if Pos.eqb x v then element else s
  | Symbol _ | Const _ => s
  end.

(* The arguments of the arms, merged: an argument that is the same in
   every arm stays, and one that is not becomes the variable. The
   elements of a table need not differ between the first two arms, so
   all the arms are gone through
   ([FlambdaMergeSwitchesSpec.merged_arguments], of each arm in turn
   with what the arms before it give). *)
Fixpoint merged_across (v : variable) (args : list simple)
    (arms : list (Z * apply_cont)) : list simple :=
  match arms with
  | [] => args
  | arm :: arms =>
    merged_across v (merged_arguments v args (ac_args (snd arm))) arms
  end.

(* An arm is the merged application with, in place of the variable, the
   element of the table that the load gives for its number. *)
Definition arm_is_loaded (v : variable) (merged : apply_cont)
    (first offset : Z) (elements : list simple) (immediates : bool)
    (arm : Z * apply_cont) : bool :=
  match table_element first offset elements (fst arm) with
  | Some element =>
    element_taken immediates element
    && (if apply_cont_eq_dec (snd arm)
             (Apply_cont_expr (ac_continuation merged)
                (map (element_for v element) (ac_args merged))
                (ac_trap_action merged))
        then true
        else false)
  | None => false
  end.

(* The variable bound again: to itself plus the offset, where the first
   arm is not that of 0, and to the element of the table at itself. The
   constant is the second argument of the addition, as Simplify has
   it. *)
Definition bind_table_offset (v : variable) (first offset : Z)
    (body : expr) : expr :=
  if first =? 0 then body
  else
    Let v Value
      (Prim (Binary (Int_arith Standard_tagged_immediate Add))
         [Var v; Const (Tagged_immediate offset)])
      body.

Definition bind_table_load (v : variable) (s : symbol) (first offset : Z)
    (immediates : bool) (body : expr) : expr :=
  bind_tagged v
    (bind_table_offset v first offset
       (Let v Value
          (Prim
             (Binary
                (Array_load (table_array_kind immediates)
                   (table_load_kind immediates) Immutable))
             [Symbol s; Var v])
          body)).

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* A table has the arguments of the arms: it has as many elements as
   there are arms, and every arm is the merged application with its
   element. *)
Definition table_has_arms (v : variable) (merged : apply_cont)
    (first offset : Z) (arms : list (Z * apply_cont))
    (table : symbol * list simple) : bool :=
  Nat.eqb (length (snd table)) (length arms)
  && (let immediates := all_immediates (snd table) in
      forallb (arm_is_loaded v merged first offset (snd table) immediates)
        arms).

Definition merge_switch_by_table (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : option expr :=
  match scrutinee, arms with
  | Var v, (first, ac) :: ((_ :: _ :: _) as others) =>
    let merged :=
      Apply_cont_expr (ac_continuation ac)
        (merged_across v (ac_args ac) others) None
    in
    if var_in_simples v (ac_args merged)
    then
      let offset := table_offset first in
      match find (table_has_arms v merged first offset arms) tables with
      | Some (s, elements) =>
        Some
          (bind_table_load v s first offset (all_immediates elements)
             (Apply_cont merged))
      | None => None
      end
    else None
  | _, _ => None
  end.

(* ======================================================================== *)
(* A way for each argument                                                  *)
(* ======================================================================== *)

(* How one argument of the arms is had where tables are in scope: in
   one of the ways of FlambdaMergeSwitchesSpec.argument_way, or as an
   element of a table. A table is kept with whether its elements are
   all immediates, as [table_way_of_column] works it out where it
   chooses the table, so that the test of an arm does not work it
   out again. *)
Inductive table_way : Type :=
  | Without_table (way : argument_way)
  | From_table (table : symbol) (elements : list simple) (immediates : bool).

Definition is_from_table (way : table_way) : bool :=
  match way with
  | From_table _ _ _ => true
  | Without_table _ => false
  end.

(* What such an argument is in the arm of number [i]: for a table, the
   element that the load gives for that number, which must be one that
   the load kind takes. *)
Definition argument_by_table_way (first offset i : Z) (way : table_way)
    : option simple :=
  match way with
  | Without_table way => argument_in_arm i way
  | From_table _ elements immediates =>
    let* element := table_element first offset elements i in
    if element_taken immediates element then Some element else None
  end.

Fixpoint arguments_by_table_ways (first offset i : Z) (ways : list table_way)
    : option (list simple) :=
  match ways with
  | [] => Some []
  | way :: ways =>
    let* s := argument_by_table_way first offset i way in
    let* args := arguments_by_table_ways first offset i ways in
    Some (s :: args)
  end.

(* An arm applies the continuation, with no trap action, to the
   arguments that the ways give for its number. *)
Definition arm_has_table_ways (k : continuation) (first offset : Z)
    (ways : list table_way) (arm : Z * apply_cont) : bool :=
  match arguments_by_table_ways first offset (fst arm) ways with
  | Some args =>
    if apply_cont_eq_dec (snd arm) (Apply_cont_expr k args None)
    then true
    else false
  | None => false
  end.

(* A table has one argument across the arms: it has as many elements
   as there are arms, and the argument of each arm is the element that
   the load gives for its number, one that the load kind takes. *)
Definition table_has_column (first offset : Z) (column : list (Z * simple))
    (table : symbol * list simple) : bool :=
  Nat.eqb (length (snd table)) (length column)
  && (let immediates := all_immediates (snd table) in
      forallb
        (fun entry =>
           match table_element first offset (snd table) (fst entry) with
           | Some element =>
             element_taken immediates element
             && (if simple_eq_dec (snd entry) element then true else false)
           | None => false
           end)
        column).

(* The way for one argument: one of the ways without a table, where
   FlambdaMergeSwitchesSpec.way_of_column finds one, as Simplify
   looks for those first; else the first table in scope that has the
   argument across the arms. *)
Definition table_way_of_column (tables : known_tables) (first offset : Z)
    (column : list (Z * simple)) : option table_way :=
  match way_of_column column with
  | Some way => Some (Without_table way)
  | None =>
    match find (table_has_column first offset column) tables with
    | Some (s, elements) =>
      Some (From_table s elements (all_immediates elements))
    | None => None
    end
  end.

Fixpoint table_ways_of_arguments (tables : known_tables) (first offset : Z)
    (count : nat) (rows : list (Z * list simple))
    : option (list table_way) :=
  match count with
  | O => Some []
  | S count =>
    let* column := column_of_arms rows in
    let* way := table_way_of_column tables first offset column in
    let* ways :=
      table_ways_of_arguments tables first offset count (rest_of_arms rows)
    in
    Some (way :: ways)
  end.

(* The variable [x] bound to the element of a table for the scrutinee
   [v], which is not bound again: to [v] tagged, then to itself plus
   the offset where the first arm is not that of 0, then to the element
   of the table at itself. *)
Definition bind_table_load_to (v x : variable) (s : symbol)
    (first offset : Z) (immediates : bool) (body : expr) : expr :=
  bind_tagged_to v x
    (bind_table_offset x first offset
       (Let x Value
          (Prim
             (Binary
                (Array_load (table_array_kind immediates)
                   (table_load_kind immediates) Immutable))
             [Symbol s; Var x])
          body)).

Definition bind_table_way (first offset : Z) (v x : variable)
    (way : table_way) (body : expr) : expr :=
  match way with
  | Without_table (Same_in_every_arm _) => body
  | Without_table (Computed how) => bind_way how v x body
  | From_table s _ immediates =>
    bind_table_load_to v x s first offset immediates body
  end.

(* The [Let]s for the arguments, the first argument outermost, each
   argument that is computed or loaded in the variable of the same
   position in [names]; and the arguments of the application under
   them. *)
Fixpoint bind_table_ways (first offset : Z) (v : variable)
    (names : list variable) (ways : list table_way) (body : expr) : expr :=
  match names, ways with
  | x :: names, way :: ways =>
    bind_table_way first offset v x way
      (bind_table_ways first offset v names ways body)
  | _, _ => body
  end.

Definition merged_by_table_way (v x : variable) (way : table_way) : simple :=
  match way with
  | Without_table (Same_in_every_arm s) => s
  | Without_table (Computed As_it_is) => Var v
  | Without_table
      (Computed (By_tagging | By_tagging_and_negating | By_arithmetic _ _ _))
  | From_table _ _ _ =>
    Var x
  end.

Fixpoint merged_by_table_ways (v : variable) (names : list variable)
    (ways : list table_way) : list simple :=
  match names, ways with
  | x :: names, way :: ways =>
    merged_by_table_way v x way :: merged_by_table_ways v names ways
  | _, _ => []
  end.

(* The names can be bound: there is one for each argument, no two are
   the same, none is the scrutinee, and none occurs in an argument
   that is the same in every arm. *)
Definition table_names_usable (v : variable) (names : list variable)
    (ways : list table_way) : bool :=
  Nat.eqb (length names) (length ways)
  && distinct names
  && negb (among v names)
  && forallb
       (fun way =>
          match way with
          | Without_table (Same_in_every_arm s) =>
            negb (existsb (fun x => var_in_simple x s) names)
          | Without_table (Computed _) | From_table _ _ _ => true
          end)
       ways.

(* The [Switch] merged with a way for each argument, one of them at
   least a table, the arguments that are computed or loaded bound to
   the parameters of the continuation that the arms apply, which is
   one of those bound around. There are three arms or more, as for a
   single table. A [Switch] none of whose arguments is loaded is left
   to FlambdaMergeSwitchesSpec.merge_switches. *)
Definition merge_switch_by_table_ways (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : option expr :=
  match scrutinee, arms with
  | Var v, (first, ac) :: _ :: _ :: _ =>
    let k := ac_continuation ac in
    let* names := lookup k known in
    let offset := table_offset first in
    let* ways :=
      table_ways_of_arguments tables first offset (length (ac_args ac))
        (map (fun arm => (fst arm, ac_args (snd arm))) arms)
    in
    if existsb is_from_table ways
       && table_names_usable v names ways
       && forallb (arm_has_table_ways k first offset ways) arms
    then
      Some
        (bind_table_ways first offset v names ways
           (Apply_cont
              (Apply_cont_expr k (merged_by_table_ways v names ways) None)))
    else None
  | _, _ => None
  end.

(* ======================================================================== *)
(* Down a term                                                              *)
(* ======================================================================== *)

(* A [Switch] merged where a table in scope has the arguments of its
   arms, if [merge_switch] leaves it: one that [merge_switch] merges is
   left to it. With no table in scope the [Switch] is left as it is,
   and nothing is computed of it. Otherwise the tables are looked at
   first: where none has the arms, [merge_switch] is not computed, and
   a way is looked for for each argument. *)
Definition merge_switch_with_tables (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) : expr :=
  match tables with
  | [] => Switch scrutinee arms
  | _ :: _ =>
    match merge_switch_by_table tables scrutinee arms with
    | Some merged =>
      match merge_switch scrutinee arms with
      | Switch _ _ => merged
      | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
      | Let_cont_recursive _ _ | Apply _ | Apply_cont _ | Invalid =>
        Switch scrutinee arms
      end
    | None =>
      match merge_switch_by_table_ways known tables scrutinee arms with
      | Some merged => merged
      | None => Switch scrutinee arms
      end
    end
  end.

(* Down a term, with the continuations that are bound around and the
   tables in scope: into the handlers of continuations, and not into
   code. A continuation is known in the body of its [Let_cont], and a
   recursive one in the handlers of its group too. *)
Fixpoint merge_tables_under (known : known_continuations)
    (tables : known_tables) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (merge_tables_under known tables body)
  | Let_static bindings body =>
    Let_static bindings
      (merge_tables_under known (tables_under_statics tables bindings) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (merge_tables_in_handler known tables h)
      (merge_tables_under ((k, parameters_of_handler h) :: known) tables body)
  | Let_cont_recursive handlers body =>
    let known :=
      map (fun entry => (fst entry, parameters_of_handler (snd entry)))
        handlers
      ++ known
    in
    Let_cont_recursive
      (map (fun '(k, h) => (k, merge_tables_in_handler known tables h))
         handlers)
      (merge_tables_under known tables body)
  | Switch scrutinee arms =>
    merge_switch_with_tables known tables scrutinee arms
  | Apply _ | Apply_cont _ | Invalid => e
  end
with merge_tables_in_handler (known : known_continuations)
    (tables : known_tables) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (merge_tables_under known tables handler)
  end.

(* That, with no continuation known at the start: what the body of a
   piece of code is given, whose return continuation is not bound in
   sight. *)
Definition merge_tables_knowing (tables : known_tables) (e : expr) : expr :=
  merge_tables_under [] tables e.

(* [merge_tables_knowing] applied to the body of every piece of code,
   with the tables in scope where the code is bound: those of the groups
   of static bindings that the binding is under and those of its own
   group. As in FlambdaNormaliseSpec.rewrite_code, the body is left
   alone if the rewrite would change whether it binds something static;
   it never does. *)
Fixpoint merge_tables_in_code (tables : known_tables) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (merge_tables_in_code tables body)
  | Let_static bindings body =>
    let tables := tables_under_statics tables bindings in
    Let_static
      (map (rewrite_code_in_binding (merge_tables_knowing tables)) bindings)
      (merge_tables_in_code tables body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (merge_tables_in_code_in_handler tables h)
      (merge_tables_in_code tables body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, merge_tables_in_code_in_handler tables h))
         handlers)
      (merge_tables_in_code tables body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with merge_tables_in_code_in_handler (tables : known_tables)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (merge_tables_in_code tables handler)
  end.

(* The whole rewrite: the term, then the bodies of its code, with no
   table known at the start. *)
Definition merge_tables (e : expr) : expr :=
  merge_tables_in_code [] (merge_tables_knowing [] e).

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The tables are bound: the static environment binds the symbol of each
   to an immutable array of the values of its elements. *)
Definition tables_bound (tables : known_tables) (statics : static_env)
    : Prop :=
  ∀ (s : symbol) (elements : list simple),
    (s, elements) ∈ tables ->
    ∃ vs : list value,
      eval_simples [] elements = Some vs
      ∧ lookup s (static_values statics) = Some (Static_immutable_array vs).

Definition merge_tables_under_refines_statement : Prop :=
  ∀ (known : known_continuations) (tables : known_tables) (e : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record),
    tables_bound tables statics ->
    refines (State e vars conts statics hp rd)
      (State (merge_tables_under known tables e) vars conts statics hp rd).

Definition merge_tables_knowing_refines_statement : Prop :=
  ∀ (tables : known_tables) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    tables_bound tables statics ->
    refines (State e vars conts statics hp rd)
      (State (merge_tables_knowing tables e) vars conts statics hp rd).

Definition merge_tables_in_code_refines_statement : Prop :=
  ∀ (tables : known_tables) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    tables_bound tables statics ->
    refines (State e vars conts statics hp rd)
      (State (merge_tables_in_code tables e) vars conts statics hp rd).

Definition merge_tables_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (merge_tables e).

Definition merge_tables_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    merge_tables (rename_expr r e) = rename_expr r (merge_tables e).

(* A term that binds no symbol to an immutable array, in its code
   either. *)
Fixpoint binds_no_table (e : expr) : bool :=
  match e with
  | Let _ _ _ body => binds_no_table body
  | Let_static bindings body =>
    forallb binding_binds_no_table bindings && binds_no_table body
  | Let_cont_non_recursive _ h body =>
    handler_binds_no_table h && binds_no_table body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_binds_no_table h) handlers
    && binds_no_table body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_binds_no_table (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => binds_no_table handler
  end
with binding_binds_no_table (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_binds_no_table c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_immutable_value_array _) => false
  | Bind_symbol _ _ => true
  end
with code_binds_no_table (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => binds_no_table body
  end.

Definition merge_tables_without_tables_statement : Prop :=
  ∀ e : expr, binds_no_table e = true -> merge_tables e = e.

(* ======================================================================== *)
(* Four terms                                                               *)
(* ======================================================================== *)

Definition scrutinee_of_example : variable := 1%positive.
Definition continuation_of_example : continuation := 1%positive.
Definition table_of_example : symbol := 1%positive.

Definition arm_of_example (n : Z) : apply_cont :=
  Apply_cont_expr continuation_of_example [Const (Tagged_immediate n)] None.

(* Four arms that give 1, 2, 3 and 0, under the binding of a table of
   those: the arms of a function from a variant of four constant
   constructors to the next one. *)
Definition switch_under_table : expr :=
  Let_static
    [Bind_symbol table_of_example
       (Static_const_immutable_value_array
          [Const (Tagged_immediate 1); Const (Tagged_immediate 2);
           Const (Tagged_immediate 3); Const (Tagged_immediate 0)])]
    (Switch (Var scrutinee_of_example)
       [(0, arm_of_example 1); (1, arm_of_example 2); (2, arm_of_example 3);
        (3, arm_of_example 0)]).

Definition load_under_table : expr :=
  Let_static
    [Bind_symbol table_of_example
       (Static_const_immutable_value_array
          [Const (Tagged_immediate 1); Const (Tagged_immediate 2);
           Const (Tagged_immediate 3); Const (Tagged_immediate 0)])]
    (Let scrutinee_of_example Value
       (Prim (Unary Tag_immediate) [Var scrutinee_of_example])
       (Let scrutinee_of_example Value
          (Prim (Binary (Array_load Array_immediates Load_immediates Immutable))
             [Symbol table_of_example; Var scrutinee_of_example])
          (Apply_cont
             (Apply_cont_expr continuation_of_example
                [Var scrutinee_of_example] None)))).

Definition merge_tables_example_statement : Prop :=
  merge_tables switch_under_table = load_under_table.

(* Three arms that give 0, 0 and 1: the first two arms are the same, and
   the argument is still loaded. *)
Definition switch_under_equal_elements : expr :=
  Let_static
    [Bind_symbol table_of_example
       (Static_const_immutable_value_array
          [Const (Tagged_immediate 0); Const (Tagged_immediate 0);
           Const (Tagged_immediate 1)])]
    (Switch (Var scrutinee_of_example)
       [(0, arm_of_example 0); (1, arm_of_example 0); (2, arm_of_example 1)]).

Definition load_under_equal_elements : expr :=
  Let_static
    [Bind_symbol table_of_example
       (Static_const_immutable_value_array
          [Const (Tagged_immediate 0); Const (Tagged_immediate 0);
           Const (Tagged_immediate 1)])]
    (Let scrutinee_of_example Value
       (Prim (Unary Tag_immediate) [Var scrutinee_of_example])
       (Let scrutinee_of_example Value
          (Prim (Binary (Array_load Array_immediates Load_immediates Immutable))
             [Symbol table_of_example; Var scrutinee_of_example])
          (Apply_cont
             (Apply_cont_expr continuation_of_example
                [Var scrutinee_of_example] None)))).

Definition merge_tables_equal_elements_example_statement : Prop :=
  merge_tables switch_under_equal_elements = load_under_equal_elements.

Definition first_parameter_of_example : variable := 2%positive.
Definition second_parameter_of_example : variable := 3%positive.
Definition return_of_example : continuation := 2%positive.
Definition second_table_of_example : symbol := 2%positive.

Definition tagged_of_example (n : Z) : simple := Const (Tagged_immediate n).

Definition pair_arm_of_example (n m : Z) : apply_cont :=
  Apply_cont_expr continuation_of_example
    [tagged_of_example n; tagged_of_example m] None.

Definition apply_parameters_of_example (k : continuation) : expr :=
  Apply_cont
    (Apply_cont_expr k
       [Var first_parameter_of_example; Var second_parameter_of_example]
       None).

(* The continuation that the arms apply, of two parameters, bound
   around a term. *)
Definition under_join_of_example (body : expr) : expr :=
  Let_cont_non_recursive continuation_of_example
    (Handler [first_parameter_of_example; second_parameter_of_example]
       (make_handler_metadata [Value; Value])
       (apply_parameters_of_example return_of_example))
    body.

(* A parameter bound to the element of a table at the scrutinee. *)
Definition load_to_of_example (x : variable) (s : symbol) (body : expr)
    : expr :=
  Let x Value (Prim (Unary Tag_immediate) [Var scrutinee_of_example])
    (Let x Value
       (Prim (Binary (Array_load Array_immediates Load_immediates Immutable))
          [Symbol s; Var x])
       body).

(* Four arms that give 1 and 10, 5 and 50, 7 and 70, 2 and 20, under
   the bindings of a table of the first numbers and of a table of the
   second ones: each parameter is loaded from its table. *)
Definition under_two_tables (body : expr) : expr :=
  Let_static
    [Bind_symbol table_of_example
       (Static_const_immutable_value_array
          [tagged_of_example 1; tagged_of_example 5; tagged_of_example 7;
           tagged_of_example 2]);
     Bind_symbol second_table_of_example
       (Static_const_immutable_value_array
          [tagged_of_example 10; tagged_of_example 50; tagged_of_example 70;
           tagged_of_example 20])]
    (under_join_of_example body).

Definition switch_under_two_tables : expr :=
  under_two_tables
    (Switch (Var scrutinee_of_example)
       [(0, pair_arm_of_example 1 10); (1, pair_arm_of_example 5 50);
        (2, pair_arm_of_example 7 70); (3, pair_arm_of_example 2 20)]).

Definition loads_under_two_tables : expr :=
  under_two_tables
    (load_to_of_example first_parameter_of_example table_of_example
       (load_to_of_example second_parameter_of_example
          second_table_of_example
          (apply_parameters_of_example continuation_of_example))).

Definition merge_tables_two_tables_example_statement : Prop :=
  merge_tables switch_under_two_tables = loads_under_two_tables.

(* Four arms that give 1 and 0, 2 and 1, 3 and 1, 4 and 0, under the
   binding of a table of the second numbers: the first parameter is
   the number of the arm, tagged, plus 1, and the second is loaded. *)
Definition under_table_beside_line (body : expr) : expr :=
  Let_static
    [Bind_symbol table_of_example
       (Static_const_immutable_value_array
          [tagged_of_example 0; tagged_of_example 1; tagged_of_example 1;
           tagged_of_example 0])]
    (under_join_of_example body).

Definition switch_under_table_beside_line : expr :=
  under_table_beside_line
    (Switch (Var scrutinee_of_example)
       [(0, pair_arm_of_example 1 0); (1, pair_arm_of_example 2 1);
        (2, pair_arm_of_example 3 1); (3, pair_arm_of_example 4 0)]).

Definition load_under_table_beside_line : expr :=
  under_table_beside_line
    (Let first_parameter_of_example Value
       (Prim (Unary Tag_immediate) [Var scrutinee_of_example])
       (Let first_parameter_of_example Value
          (Prim (Binary (Int_arith Standard_tagged_immediate Add))
             [tagged_of_example 1; Var first_parameter_of_example])
          (load_to_of_example second_parameter_of_example table_of_example
             (apply_parameters_of_example continuation_of_example)))).

Definition merge_tables_table_and_line_example_statement : Prop :=
  merge_tables switch_under_table_beside_line = load_under_table_beside_line.
