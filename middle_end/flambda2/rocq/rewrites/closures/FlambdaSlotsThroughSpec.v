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

(* A value slot that is read through another closure: in the body of
   code, the projection of a value slot from the closure that
   [my_closure] holds in one of its value slots becomes the projection
   of a value slot of [my_closure] itself, where every set that has
   that code was given the two from one simple. The same is done of a
   closure that the body reads from a value slot and knows the code
   of, in the place of [my_closure].

   This file is to be reviewed: [made_under_let] and what it is made
   of, which say what is in sight and what is forgotten; [sets_through]
   and [sets_through_give], the test; [state_slots_through], what the
   test gives of a state; [read_through] and [through_under_let], the
   rewrite of one defining expression and what it remembers;
   [read_slots_through]; [know_own_closures_through], which ends the
   Rocq round FlambdaValidateSpec.normalise_lifted_round_own where
   [own] is set; and the statements.

   What the source has. A functor makes, each time it is applied, a
   closure [u]; a closure [t] that captures [u]; and a closure [w]
   that captures both [t] and [u]. The code of [w] calls [t], and the
   code of [t] reads its [u]. In stdlib/set.ml they are [union],
   [try_concat] and [filter_map]; in stdlib/pqueue.ml [t] is the
   comparison [lt] of a functor and [u] the argument of the functor.

   What Simplify does. Where it inlines the code of [t] into the code
   of [w], the read of [u] in the copy is a projection of a value slot
   from the closure that [w] holds in a value slot. Simplify makes it
   the read of the value slot of [w] itself that holds [u]. It
   simplifies the code of a set with the types of what the set puts in
   its value slots (simplify/simplify_set_of_closures.ml,
   [dacc_inside_function]; FlambdaClosureOfCodeSpec.v says the same of
   the code of the function slots). How it comes to the read of the
   other value slot has not been followed through the compiler line
   by line. It seems to be this: the variables of the place where the
   set is made stay in the environment of the code at a name mode
   that only types may use; the type of the value slot that holds [t]
   says that the value slot of [t] is that variable [u], the type of
   the value slot that holds [u] says the same, and the first read is
   replaced by the variable that the second is bound to. What is
   certain is the two terms, in which the term before reads through
   the other closure and the term after reads the closure's own value
   slot.

   The fact. For a code [c] at a function slot [f], two value slots
   [a] and [b] of its sets and a value slot [d]: every set of the unit
   that has [c] at [f] gives its value slot [a] a variable in sight
   that a [Let] bound to a closure which that [Let] makes, and that
   [Let] gave the value slot [d] of its set the very simple that the
   first set gives its own value slot [b] ([sets_through_give]). The
   simple is a constant, a symbol or a variable; a variable is one
   that has not been bound again between the two places, by what is
   forgotten below. Nothing is guessed: the test is of the sets that
   the unit makes, by a [Let] or statically, in handlers and in code
   too ([sets_through]).

   What is in sight ([made_sight]): the variables that a [Let] above
   bound to a closure that the [Let] makes ([Set_of_closures],
   [Closure_of_set]), each with what the [Let] gave the value slots of
   its set ([slots_made]). It is the sight of FlambdaSlotsOfCodeSpec.v
   with another thing said of a variable, and with one case fewer: a
   variable bound to the projection of a function slot from a variable
   in sight is in sight there and is not here. Such a closure has the
   value slots of the set that it is projected from, so it could be;
   it is left out to keep what is forgotten simple.

   What is forgotten ([made_without], [made_under_let]). Where a
   variable [x] is bound, by a [Let] or as a parameter of a handler:
   [x] is no longer in sight; and of every other variable in sight,
   each value slot that was given [Var x] is no longer known, as
   [Var x] is now another value. The slot stays listed, with nothing
   known of it: were it taken out, a later slot of the same name in
   the list would be read in its place, and the closure does not hold
   that one. The [Let] that makes a closure binds its own variable,
   so the same is done to what it has just given: in
   [Let x = closure with slot d given Var x], the slot was given the
   old [x] and is not known of the new one. The handler of a
   continuation runs under the variables of the place where it is
   defined, so what is in sight there, less its parameters and what
   they make forgotten, is in sight in it; the body of code runs under
   its parameters and [my_closure] alone, so nothing is in sight where
   it starts.

   What the test gives of a run ([state_slots_through]). Every closure
   of a state that is of a set with [c] at [f] holds in its value slot
   [a] a closure value, which is no symbol, and that closure value
   holds in its value slot [d] the value that the first closure holds
   in its own value slot [b] ([set_held_through]). A closure value is
   not changed once it is made, and the two value slots were filled
   from one simple with nothing bound between that the simple names;
   so the two hold one value, be it a number, a closure or a
   location. The cell that a location names may be written; the
   location in the two value slots is the same location. The state
   that a unit starts in has it where the unit passes the test
   ([state_slots_through_initial_statement]); a step keeps it
   ([state_slots_through_step_statement]); and in a state that has
   it, the two projections give one value
   ([closure_slot_through_statement],
   [projection_through_statement]).

   The slots are read as the machine reads them: every look-up of a
   function slot or of a value slot, in the test and in what the test
   gives, is the first binding of the name ([lookup]), as
   FlambdaValueSpec.closure_in_set and
   FlambdaValueSpec.project_value_slot read. A unit is not asked to
   name a slot once.

   The rewrite ([read_through], in [know_through_in_body]). In the
   body of code that is no stub and whose [my_closure] is not also a
   parameter, of a variable [x] known to be a closure of a code [c]
   (as the second pass of FlambdaClosureOfCodeSpec.know_own_closures
   knows it, FlambdaClosureOfCodeSpec.codes_under_read:
   [my_closure]; a function slot projected from a variable that is
   known; and a value slot projected from a variable that is known,
   where every set with the code of that variable gives the slot a
   closure in sight of one code, the fact of
   FlambdaSlotsOfCodeSpec.v): a
   variable [y] bound to [Project_value_slot_from f a (Var x)] is
   remembered as that ([through_under_let]), down to where [x] or [y]
   is bound again; and [Project_value_slot_from g d (Var y)] becomes
   [Project_value_slot_from f b (Var x)], where [b] is a value slot
   for which the sets of the unit pass the test at [c], [f], [a], [b]
   and [d] ([slot_through]). The function slot [g] is not looked at:
   in a run that comes to the projection that is replaced, where
   that projection has a value, its closure is at [g], and the value
   is that of the projection put in its place
   ([projection_through_statement]); if a run comes to it and it has
   no value, the run is undefined there, and nothing is said of the
   run of the unit rewritten.
   Nothing else changes: the projection of [a] stays, with its
   [Let] and the kind of its [Let]; the [Let] of the projection that
   is replaced keeps its kind; and no projection without a function
   slot ([Project_value_slot]) is read or made.

   So the closure read from need not be [my_closure]. In
   stdlib/pqueue.ml, with -O3, the closure of [pop_min] holds the
   closure of [sift_down] in a value slot, and Simplify inlines one
   turn of [sift_down] into the code of [pop_min]: the copy reads the
   closure of [lt] from the closure of [sift_down], and the argument
   of the functor through the closure of [lt]. Simplify reads the
   argument from the closure of [sift_down], which holds it too. In
   the code of [pop_min], [x] is then the variable bound to the
   closure of [sift_down], which is known by the fact of
   FlambdaSlotsOfCodeSpec.v, and [c] is the code of [sift_down]. The
   two value slots are those of the sets with [c], and the test is
   the same test; the read put in place is of [x].

   What is remembered is forgotten as the sight is
   ([through_without]): where a variable is bound, what was
   remembered of it goes, and so does what was remembered as
   projected from it. So [Let x = Project_value_slot_from f a (Var x)]
   is not remembered: the [x] that it binds is not the [x] that it
   reads. The projection put in place names [x], which the one
   replaced did not: [x] is the variable that the first projection
   read, and it has not been bound between the first projection and
   the place of the second, or the second would not be replaced. That
   is why the result has no free variable that the unit has not.

   The law is that of FlambdaClosureOfCodeSpec.know_own_closures. The
   unit rewritten refines the unit run on its own, from its imports,
   and followed by what may follow a unit, which is the observers of
   FlambdaObserversSpec.v run from the imports (the three statements
   of [read_slots_through]). What makes it right is that every closure
   of the run of a unit is of a set that the unit makes, and that the
   body of code runs only where [my_closure] is a closure of that code
   (FlambdaContextWhereSpec.let_static_code_refines_from_called_statement).
   A variable that is known by a value slot is bound to a closure of
   its code in the same runs
   (FlambdaSlotsOfCodeSpec.closure_slot_gives_statement, with the
   sets as FlambdaSlotsOfCodeSpec.sets_seen sees them).
   As there, the rewrite is made only of a unit whose groups bind
   their static sets whole
   (FlambdaClosureOfCodeSpec.groups_whole), which is what the code
   known of a projected function slot rests on.

   What it is not.
   - Not under all environments, and not of a unit followed by any
     term: a closure from elsewhere may have [c] at [f] with anything
     in its value slots. A later unit that made sets with a code id of
     this unit is not among what may follow, and the heap that the run
     starts with is empty.
   - Not of a closure that was lifted into [a]: where a set gives its
     value slot [a] a symbol, nothing is known by this fact, whatever
     the symbol is bound to (FlambdaSlotsOfCodeSpec.v says why of its
     own fact). It is the closure held in [a] that must be no symbol.
     The closure that holds it, which the body reads as [x], may be a
     symbol bound to a static set: the test looks at the static sets
     too ([closure_slot_through_statement]).
   - Not of a closure in [a] that the set did not get from a [Let]
     which made it and which is in sight: a parameter, a variable
     bound to another variable, to the projection of a function slot
     or to anything else is not in sight, whatever it holds when the
     unit runs.
   - Not of an alias: the two value slots must be given the same
     simple as it is written. Where one is given a variable and the
     other a second variable bound to the first, the test fails.
   - Not of a copy of the code under another code id.
   - Not from a closure [x] that is not known. A variable bound to a
     value slot of a variable that is known is known only where
     every set with the code of the second gives the slot a closure
     in sight of one code: not where one of them gives it a symbol,
     a parameter, a variable bound to another variable, or closures
     of two codes (FlambdaSlotsOfCodeSpec.v says what is in sight
     there).
   - One step through one closure: the read that is replaced is of a
     value slot of the closure [y], and what is put in its place is a
     read of the closure [x] that [y] was read from. Where [x] was
     itself read from a value slot of [my_closure], the read put in
     place is of [x] and not of [my_closure]. Each replacement uses
     the variable that its first projection read; one use of
     [read_slots_through] may replace several reads, and what is known
     of the variables may come through several value slots, but it
     does not rewrite what it has put in place to follow a chain. A
     value slot read through two closures is not read from the first
     by one use of [read_slots_through], and nothing here says how
     often it is to be used.
   - The two value slots of one closure hold one value. Nothing is
     said of two closures: the common simple is of each place where a
     set is made, and need not be one simple, or one value, for all
     the sets with [c] at [f], or for all the times that one place
     runs. The code of the closure in [a] need not be one code either.
   - Nothing of what a cell holds. Where the common value is a
     location, the two projections give that location; what is read
     from it later is whatever the cell then holds.
   - No identity of closures (decision D2), and nothing of an answer
     of [Phys_equal].
   - Nothing here says that a projection has a value. The machine
     makes a projection that names a function slot its closure is not
     at, a projection of a value slot that its closure has not got,
     and a [Let] whose kind its value has not, undefined, as before.
     The rewrite does not promise to keep what is undefined
     undefined: it may put, in the place of a projection that names
     the wrong function slot, one that has a value. The equality that
     its refinement uses supposes that the projection replaced has a
     value. The [Let]s and their kinds are kept; where the test finds
     no fact the projection is left as it is.
   - A rewrite of the term before, which refines. Nothing of it is
     applied to the term after, and it is no equivalence: the unit
     rewritten may be defined where the unit is not.
   - Of the machine of FlambdaSemanticsSpec.v only, which is the one
     that runs a term with its regions erased
     (FlambdaRegionsErasureSpec.v, under what that file asks). Nothing
     is said of the machine with regions, of its checks or of
     lifetimes: that two value slots hold one value does not say that
     the value may be read from either, where one of the closures is
     in a region that has ended.
   - What an exit observes of a closure (decision D1) is not touched:
     the rewrite changes the body of code and no value slot. Nothing
     is added to what an exit carries (decision D57).
   - Not a claim that the validation of any function or file passes
     with it.

   The result has its code closed where the unit has, and no free
   variable or continuation that the unit has not
   ([read_slots_through_names_statement]); and the rewrite commutes
   with renaming ([read_slots_through_equivariant_statement]).

   What the validation applies ([know_own_closures_through]) is
   FlambdaClosureOfCodeSpec.know_own_closures and then
   [read_slots_through], of what the first gives: the sets of the test
   are those that the term then makes. Its five laws are the laws of
   the two, composed. It is here and not a part of
   [know_own_closures] because this file is built on that one.

   The examples at the end are each one unit and what
   [read_slots_through] gives of it: the reads that are replaced, and
   the ones that must not be.

   The proofs are in FlambdaSlotsThroughProofs.v, which needs no
   review. The proofs are of the definitions as they are written
   here: that the test, the sight and the rewrite are the ones meant
   is what the review of this file is for. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaBindersSpec FlambdaNominalSpec
  FlambdaNormaliseSpec FlambdaUnusedStaticsSpec FlambdaKnownSlotsSpec
  FlambdaSetsOfCodeSpec FlambdaSlotsOfCodeSpec FlambdaContextWhereSpec
  FlambdaClosureOfCodeSpec.
From OxCamlFlambda2 Require Import FlambdaImportsSpec FlambdaObserversSpec.
Import ListNotations.

(* ======================================================================== *)
(* The closures in sight, with what their value slots were given            *)
(* ======================================================================== *)

(* A closure that a [Let] made, as it is seen where it is in sight:
   the value slots of its set as the [Let] listed them, each with the
   simple that the [Let] gave it, or with nothing where that has been
   forgotten. A slot that is forgotten keeps its place in the list: a
   slot is read as the machine reads it, by the first of its name, and
   a later one of the same name is not what the closure holds. *)
Definition made_seen : Type := list (value_slot * option simple).

(* The variables in sight that a [Let] bound to a closure that it
   made. *)
Definition made_sight : Type := finite_map made_seen.

(* What a defining expression gives the value slots of the set that it
   makes, if it makes one. The projection of a function slot makes no
   set. *)
Definition slots_made (d : named) : option made_seen :=
  match d with
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    Some (map_values Some value_slots)
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    None
  end.

(* What is still known of a value slot once [x] is bound again:
   nothing, if it was given [Var x]. *)
Definition forget_simple (x : variable) (given : option simple)
    : option simple :=
  match given with
  | Some s => if var_in_simple x s then None else Some s
  | None => None
  end.

(* What is still known of a closure once [x] is bound again: nothing
   of the value slots that were given [Var x]. *)
Definition forget_in_made (x : variable) (made : made_seen) : made_seen :=
  map_values (forget_simple x) made.

(* What is in sight once [x] is bound again: not [x], and of the
   others nothing of the value slots that were given [Var x]. *)
Definition made_without (x : variable) (known : made_sight) : made_sight :=
  map_values (forget_in_made x) (unbind x known).

(* The same of several variables, bound one after the other. *)
Fixpoint made_without_all (xs : list variable) (known : made_sight)
    : made_sight :=
  match xs with
  | [] => known
  | x :: xs => made_without_all xs (made_without x known)
  end.

(* What is in sight in the body of a [Let] of [x] to [d]: what was,
   once [x] is bound again; and [x], if [d] makes a set, with what [d]
   gave the value slots of that set less those that it gave the old
   [x]. *)
Definition made_under_let (known : made_sight) (x : variable) (d : named)
    : made_sight :=
  match slots_made d with
  | Some made => (x, forget_in_made x made) :: made_without x known
  | None => made_without x known
  end.

(* ======================================================================== *)
(* The sets of closures that a term makes, as they are seen                 *)
(* ======================================================================== *)

(* What a set gives a value slot, as it is seen where the set is made:
   the simple and, if that is a variable in sight, what is known of
   the value slots of the closure that it is bound to. *)
Definition slot_through_seen : Type := (simple * option made_seen)%type.

Definition simple_made_seen (known : made_sight) (s : simple)
    : option made_seen :=
  match s with
  | Var y => lookup y known
  | Symbol _ | Const _ => None
  end.

(* A set of closures as a term makes it and as it is seen there: the
   code of its function slots and what it gives each of its value
   slots. *)
Definition set_through : Type :=
  (list (function_slot * code_id)
   * list (value_slot * slot_through_seen))%type.

Definition slots_through (known : made_sight)
    (value_slots : list (value_slot * simple))
    : list (value_slot * slot_through_seen) :=
  map (fun '(slot, s) => (slot, (s, simple_made_seen known s))) value_slots.

(* The set that a defining expression, or a static constant, makes
   (FlambdaSetsOfCodeSpec.named_sets_made,
   FlambdaSetsOfCodeSpec.static_const_sets_made), seen with what is in
   sight there. *)
Definition named_sets_through (known : made_sight) (d : named)
    : list set_through :=
  match d with
  | Set_of_closures code value_slots =>
    [([(unnamed_function_slot, code)], slots_through known value_slots)]
  | Closure_of_set _ function_decls value_slots =>
    [(function_decls, slots_through known value_slots)]
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    []
  end.

Definition static_const_sets_through (known : made_sight)
    (const : static_const) : list set_through :=
  match const with
  | Static_const_set_of_closures code value_slots =>
    [([(unnamed_function_slot, code)], slots_through known value_slots)]
  | Static_const_closure_of_set _ _ function_decls value_slots =>
    [(function_decls, slots_through known value_slots)]
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    []
  end.

(* All the sets that an expression makes, in handlers and in code too
   (FlambdaSetsOfCodeSpec.sets_made), each seen with what is in sight
   where it is made: under a [Let], what [made_under_let] gives; in a
   handler, what is in sight where the handler is defined once its
   parameters are bound; in the body of code, nothing to begin
   with. The set that a [Let] makes is seen with what is in sight
   above that [Let]. *)
Fixpoint sets_through (known : made_sight) (e : expr) : list set_through :=
  match e with
  | Let x _ d body =>
    named_sets_through known d ++ sets_through (made_under_let known x d) body
  | Let_static bindings body =>
    flat_map (binding_sets_through known) bindings ++ sets_through known body
  | Let_cont_non_recursive _ h body =>
    handler_sets_through known h ++ sets_through known body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_sets_through known h) handlers
    ++ sets_through known body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_sets_through (known : made_sight) (h : continuation_handler)
    : list set_through :=
  match h with
  | Handler params _ handler =>
    sets_through (made_without_all params known) handler
  end
with binding_sets_through (known : made_sight) (binding : static_binding)
    : list set_through :=
  match binding with
  | Bind_code _ c => code_sets_through c
  | Bind_deleted_code _ => []
  | Bind_symbol _ const => static_const_sets_through known const
  end
with code_sets_through (c : code) : list set_through :=
  match c with
  | Code _ _ _ _ _ body => sets_through [] body
  end.

(* ======================================================================== *)
(* The test                                                                 *)
(* ======================================================================== *)

(* A set, as it is seen, gives the value slot [a] a variable in sight
   whose [Let] gave the value slot [d] of its set the simple that this
   set gives its own value slot [b], and that is not forgotten. Each
   value slot is the first of its name. *)
Definition set_through_gives (a b d : value_slot) (set : set_through)
    : bool :=
  match lookup a (snd set), lookup b (snd set) with
  | Some (_, Some made), Some (held, _) =>
    match lookup d made with
    | Some (Some inner) => same_simple inner held
    | Some None | None => false
    end
  | _, _ => false
  end.

(* Each of the sets that has [c] at [f] does. *)
Definition sets_through_give (c : code_id) (f : function_slot)
    (a b d : value_slot) (seen : list set_through) : bool :=
  forallb
    (fun set =>
       negb (has_code_at c f (fst set)) || set_through_gives a b d set)
    seen.

(* ======================================================================== *)
(* What the test gives of a state                                           *)
(* ======================================================================== *)

(* The values of the value slots of a closure value. A symbol is not a
   closure value, whatever it is bound to. *)
Definition closure_value_slots (v : value)
    : option (list (value_slot * value)) :=
  match v with
  | Value_closure _ slot_values | Value_closure_of_set _ _ slot_values =>
    Some slot_values
  | Value_const _ | Value_symbol _ | Value_block _ _
  | Value_immutable_array _ | Value_boxed_number _ | Value_location _
  | Value_partial_application _ _ _ =>
    None
  end.

(* What is in sight holds of the variables: each variable in sight is
   bound to a closure value, and each value slot that is known of it
   holds the value that the simple it was given has under the
   variables now. *)
Definition made_sight_holds (known : made_sight) (vars : env) : Prop :=
  ∀ (y : variable) (made : made_seen),
    lookup y known = Some made ->
    ∃ (v : value) (slot_values : list (value_slot * value)),
      lookup y vars = Some v
      ∧ closure_value_slots v = Some slot_values
      ∧ ∀ (slot : value_slot) (s : simple),
          lookup slot made = Some (Some s) ->
          ∃ w : value,
            lookup slot slot_values = Some w ∧ eval_simple vars s = Some w.

(* A set that a closure is of, with the values of its value slots, if
   it has [c] at [f]: it holds in [a] a closure value, which holds in
   [d] the value that the set holds in [b]. Nothing is asked of
   another set. *)
Definition set_held_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) : Prop :=
  has_code_at c f function_decls = true ->
  ∃ (inner : value) (inner_slots : list (value_slot * value)) (w : value),
    lookup a slot_values = Some inner
    ∧ closure_value_slots inner = Some inner_slots
    ∧ lookup d inner_slots = Some w
    ∧ lookup b slot_values = Some w.

(* Every closure in the value is of a set that is so, through the
   values of value slots, the fields of blocks and partial
   applications (as FlambdaSlotsOfCodeSpec.value_slots_give). A symbol
   is so whatever it is bound to: what it is bound to is asked to be
   so where it is bound ([allocated_slots_through]). *)
Fixpoint value_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (v : value) : Prop :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _ | Value_location _ =>
    True
  | Value_closure code slot_values =>
    set_held_through c f a b d [(unnamed_function_slot, code)] slot_values
    ∧ fold_right
        (fun contents acc =>
           value_slots_through c f a b d (snd contents) ∧ acc)
        True slot_values
  | Value_closure_of_set _ function_decls slot_values =>
    set_held_through c f a b d function_decls slot_values
    ∧ fold_right
        (fun contents acc =>
           value_slots_through c f a b d (snd contents) ∧ acc)
        True slot_values
  | Value_block _ fields | Value_immutable_array fields =>
    fold_right (fun field acc => value_slots_through c f a b d field ∧ acc)
      True fields
  | Value_partial_application callee args _ =>
    value_slots_through c f a b d callee
    ∧ fold_right (fun arg acc => value_slots_through c f a b d arg ∧ acc)
        True args
  end.

Definition values_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (vs : list value) : Prop :=
  fold_right (fun v acc => value_slots_through c f a b d v ∧ acc) True vs.

Definition allocated_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (allocated : statically_allocated_value) : Prop :=
  match allocated with
  | Static_closure code slot_values =>
    set_held_through c f a b d [(unnamed_function_slot, code)] slot_values
    ∧ values_slots_through c f a b d (map snd slot_values)
  | Static_closure_of_set _ _ function_decls slot_values =>
    set_held_through c f a b d function_decls slot_values
    ∧ values_slots_through c f a b d (map snd slot_values)
  | Static_block _ fields => values_slots_through c f a b d fields
  | Static_immutable_array elements =>
    values_slots_through c f a b d elements
  | Static_boxed_number contents => value_slots_through c f a b d contents
  | Static_string _ | Static_imported _ => True
  end.

(* An expression that runs under the variables [vars] makes sets that
   pass the test, seen under some sight that holds of [vars]. *)
Definition expr_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (vars : env) (e : expr) : Prop :=
  ∃ known : made_sight,
    made_sight_holds known vars
    ∧ sets_through_give c f a b d (sets_through known e) = true.

(* The handlers that a continuation keeps make sets that pass the
   test, seen under some sight that holds of the variables that the
   continuation keeps; and so of the continuations that it keeps (as
   FlambdaSlotsOfCodeSpec.cont_slots_give). *)
Fixpoint cont_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (k : cont_value) : Prop :=
  match k with
  | Outer _ => True
  | Closure h vars conts recursive =>
    (∃ known : made_sight,
       made_sight_holds known vars
       ∧ sets_through_give c f a b d (handler_sets_through known h) = true
       ∧ forallb
           (fun '(_, h) =>
              sets_through_give c f a b d (handler_sets_through known h))
           recursive
         = true)
    ∧ fold_right
        (fun kc acc => cont_slots_through c f a b d (snd kc) ∧ acc)
        True conts
  | Extra_args k' _ => cont_slots_through c f a b d k'
  | Over_application _ _ _ return_to exn_to =>
    cont_slots_through c f a b d return_to
    ∧ cont_slots_through c f a b d exn_to
  end.

Definition conts_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (conts : cont_env) : Prop :=
  fold_right (fun kc acc => cont_slots_through c f a b d (snd kc) ∧ acc) True
    conts.

(* The code that is bound makes sets that pass the test, seen with
   nothing in sight where its body starts. *)
Definition code_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (statics : static_env) : Prop :=
  ∀ (id : code_id) (code : code),
    lookup id (static_code statics) = Some code ->
    sets_through_give c f a b d (code_sets_through code) = true.

(* Every set that the state holds or may come to make, and that has
   [c] at [f], holds in [a] a closure value that holds in [d] what the
   set holds in [b]: those that its terms make (its expression, the
   handlers that its continuations keep, the code that is bound), and
   those of the closures in its values (those of its variables, of
   the environments that its continuations keep and of the cells of
   its heap, and what each symbol is bound to;
   FlambdaSemanticsSpec.state_values_all). *)
Definition state_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (s : state) : Prop :=
  expr_slots_through c f a b d (state_vars s) (state_expr s)
  ∧ conts_slots_through c f a b d (state_conts s)
  ∧ code_slots_through c f a b d (state_statics s)
  ∧ state_values_all (fun _ v => value_slots_through c f a b d v)
      (fun _ allocated => allocated_slots_through c f a b d allocated) s.

(* ======================================================================== *)
(* The rewrite of the body of code                                          *)
(* ======================================================================== *)

(* The value slot [b] of the sets with [c] at [f] that is given what
   the closure in their value slot [a] was given for its value slot
   [d]: the first value slot of the first such set for which every
   such set passes the test, if there is such a set and such a value
   slot. *)
Definition slot_through (seen : list set_through) (c : code_id)
    (f : function_slot) (a d : value_slot) : option value_slot :=
  match find (fun set => has_code_at c f (fst set)) seen with
  | Some set =>
    option_map fst
      (find (fun entry => sets_through_give c f a (fst entry) d seen)
         (snd set))
  | None => None
  end.

(* The variables that a [Let] bound to the projection of a value slot
   from a variable: the variable projected from, the function slot
   that the projection names and the value slot. *)
Definition projections : Type :=
  finite_map (variable * function_slot * value_slot).

(* What is remembered once [x] is bound again: nothing of [x], and
   nothing that was projected from the old [x]. *)
Definition through_without (x : variable) (through : projections)
    : projections :=
  filter (fun entry => negb (Pos.eqb x (fst (fst (snd entry)))))
    (unbind x through).

Fixpoint through_without_all (xs : list variable) (through : projections)
    : projections :=
  match xs with
  | [] => through
  | x :: xs => through_without_all xs (through_without x through)
  end.

(* What is remembered in the body of a [Let] of [x] to [d]: what was,
   once [x] is bound again; and [x], if [d] is the projection of a
   value slot from another variable. The projection of a value slot
   of [x] from [x] itself is not remembered: the [x] bound is not the
   [x] read. *)
Definition through_under_let (through : projections) (x : variable)
    (d : named) : projections :=
  match d with
  | Project_value_slot_from f a (Var from) =>
    if Pos.eqb from x then through_without x through
    else (x, (from, f, a)) :: through_without x through
  | Project_value_slot_from _ _ (Symbol _ | Const _) | Simple _ | Prim _ _
  | Set_of_closures _ _ | Closure_of_set _ _ _ | Project_value_slot _ _
  | Project_function_slot _ _ _ =>
    through_without x through
  end.

(* The projection of the value slot [d] from a variable remembered as
   the value slot [a] of a variable known to be a closure of the code
   [c] is the projection, from that variable, of the value slot that
   every set with [c] at the function slot of the first projection
   gives what the closure in [a] was given for [d]. *)
Definition read_through (seen : list set_through) (known : closure_codes)
    (through : projections) (d : named) : named :=
  match d with
  | Project_value_slot_from _ slot (Var y) =>
    match lookup y through with
    | Some (x, f, a) =>
      match lookup x known with
      | Some c =>
        match slot_through seen c f a slot with
        | Some b => Project_value_slot_from f b (Var x)
        | None => d
        end
      | None => d
      end
    | None => d
    end
  | Project_value_slot_from _ _ (Symbol _ | Const _) | Simple _ | Prim _ _
  | Set_of_closures _ _ | Closure_of_set _ _ _ | Project_value_slot _ _
  | Project_function_slot _ _ _ =>
    d
  end.

(* The body of code. What is known of the variables is what the
   second pass of FlambdaClosureOfCodeSpec.know_own_closures knows
   (FlambdaClosureOfCodeSpec.codes_under_read, with the sets [codes]
   as FlambdaSlotsOfCodeSpec.sets_seen sees them), from the defining
   expressions as they were: [my_closure], a function slot projected
   from a variable that is known, and a value slot projected from a
   variable that is known which every set with its code gives a
   closure in sight of one code. Only the defining expression of a
   [Let] is rewritten. *)
Fixpoint know_through_in_body (sets : list set_made) (codes : list set_seen)
    (seen : list set_through) (known : closure_codes)
    (through : projections) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (read_through seen known through d)
      (know_through_in_body sets codes seen
         (codes_under_read sets codes known x d)
         (through_under_let through x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (know_through_in_body sets codes seen known through body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (know_through_in_handler sets codes seen known through h)
      (know_through_in_body sets codes seen known through body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, know_through_in_handler sets codes seen known through h))
         handlers)
      (know_through_in_body sets codes seen known through body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with know_through_in_handler (sets : list set_made) (codes : list set_seen)
    (seen : list set_through) (known : closure_codes)
    (through : projections) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (know_through_in_body sets codes seen (forget_code_vars params known)
         (through_without_all params through) handler)
  end.

(* The code [id], as FlambdaClosureOfCodeSpec.know_slots_in_code: in
   its body, [my_closure] is a closure of [id], unless the code is that
   of a stub or [my_closure] is also one of its parameters. Nothing is
   remembered where the body starts. *)
Definition know_through_in_code (sets : list set_made)
    (codes : list set_seen) (seen : list set_through) (id : code_id)
    (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    if cm_is_stub metadata || among my_closure params then c
    else
      Code params my_closure ret exn metadata
        (know_through_in_body sets codes seen [(my_closure, id)] [] body)
  end.

Definition know_through_in_binding (sets : list set_made)
    (codes : list set_seen) (seen : list set_through)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (know_through_in_code sets codes seen id c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

(* Every code that the expression binds, in handlers too; not the code
   that the body of code binds. *)
Fixpoint know_code_through (sets : list set_made) (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (know_code_through sets codes seen body)
  | Let_static bindings body =>
    Let_static (map (know_through_in_binding sets codes seen) bindings)
      (know_code_through sets codes seen body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (know_code_through_in_handler sets codes seen h)
      (know_code_through sets codes seen body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) => (k, know_code_through_in_handler sets codes seen h))
         handlers)
      (know_code_through sets codes seen body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with know_code_through_in_handler (sets : list set_made)
    (codes : list set_seen) (seen : list set_through)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (know_code_through sets codes seen handler)
  end.

(* The unit with the value slots that are read through another closure
   read from the closure that the other was read from, from the sets
   that the unit makes: as made, as FlambdaSlotsOfCodeSpec.sets_seen
   sees them from nothing in sight (for the code of a closure that a
   value slot holds), and as [sets_through] sees them from nothing in
   sight (for the test); where its groups bind their static sets
   whole. The unit as it is otherwise. *)
Definition read_slots_through (e : expr) : expr :=
  if groups_whole e
  then know_code_through (sets_made e) (sets_seen [] e) (sets_through [] e) e
  else e.

(* ======================================================================== *)
(* The statements: the fact                                                 *)
(* ======================================================================== *)

(* The state that a unit starts in, run on its own, has the fact where
   the sets that the unit makes pass the test, seen from nothing in
   sight. Nothing else is asked of the unit. *)
Definition state_slots_through_initial_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (a b d : value_slot) (e : expr),
    sets_through_give c f a b d (sets_through [] e) = true ->
    state_slots_through c f a b d
      (State e [] [] empty_static_env empty_heap no_reads).

(* The same from a static environment that binds something already and
   with continuations: the code that the static environment has and
   the handlers that the continuations keep make sets that pass the
   test, and the values that they hold have the fact. The heap is
   empty. *)
Definition state_slots_through_from_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (a b d : value_slot) (e : expr)
    (conts : cont_env) (statics : static_env),
    sets_through_give c f a b d (sets_through [] e) = true ->
    conts_slots_through c f a b d conts ->
    conts_values_all (value_slots_through c f a b d) conts ->
    code_slots_through c f a b d statics ->
    Forall
      (fun binding => allocated_slots_through c f a b d (snd binding))
      (static_values statics) ->
    state_slots_through c f a b d
      (State e [] conts statics empty_heap no_reads).

(* A step keeps it. *)
Definition state_slots_through_step_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (a b d : value_slot),
    kept_by_step (state_slots_through c f a b d).

(* What is in sight is kept by a [Let]: where it holds of the
   variables and the defining expression has a value, what
   [made_under_let] gives holds of the variables with the variable of
   the [Let] bound to that value, whether or not the variable was
   bound before and whatever was given the old one. *)
Definition made_sight_holds_under_let_statement : Prop :=
  ∀ (statics : static_env) (known : made_sight) (vars : env) (x : variable)
    (d : named) (v : value),
    made_sight_holds known vars ->
    eval_named statics vars d = Some v ->
    made_sight_holds (made_under_let known x d) ((x, v) :: vars).

(* In a state that has the fact, a closure that a variable is bound
   to, at the function slot [f] of a set that has [c] there, holds in
   [a] a closure value that holds in [d] the value that the closure
   holds in [b]. The closure may be a symbol; what it holds in [a] is
   not. *)
Definition closure_slot_through_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (a b d : value_slot) (s : state)
    (x : variable) (v : value)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)),
    state_slots_through c f a b d s ->
    lookup x (state_vars s) = Some v ->
    closure_in_set (state_statics s) v
    = Some (f, function_decls, slot_values) ->
    lookup f function_decls = Some c ->
    ∃ (inner : value) (inner_slots : list (value_slot * value)) (w : value),
      lookup a slot_values = Some inner
      ∧ closure_value_slots inner = Some inner_slots
      ∧ lookup d inner_slots = Some w
      ∧ lookup b slot_values = Some w.

(* So the two projections give one value: in a state that has the
   fact, where a variable is bound to a closure of the code [c] at
   [f], the projection of [a] from it has the value [inner], and the
   projection of [d] from [inner], naming whatever function slot, has
   the value [w], the projection of [b] from the closure has the value
   [w]. *)
Definition projection_through_statement : Prop :=
  ∀ (c : code_id) (f g : function_slot) (a b d : value_slot) (s : state)
    (x : variable) (v inner w : value)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)),
    state_slots_through c f a b d s ->
    lookup x (state_vars s) = Some v ->
    closure_in_set (state_statics s) v
    = Some (f, function_decls, slot_values) ->
    lookup f function_decls = Some c ->
    project_value_slot (state_statics s) f a v = Some inner ->
    project_value_slot (state_statics s) g d inner = Some w ->
    project_value_slot (state_statics s) f b v = Some w.

(* ======================================================================== *)
(* The statements: the rewrite                                              *)
(* ======================================================================== *)

(* A unit whose groups bind their static sets whole, with the value
   slots read through another closure read from the closure that the
   other was read from, from sets that include those that it makes,
   as made and as seen in the two ways from nothing in sight, refines
   the unit, run on its own. *)
Definition know_code_through_program_refines_statement : Prop :=
  ∀ (sets : list set_made) (codes : list set_seen) (seen : list set_through)
    (e : expr),
    incl (sets_made e) sets ->
    incl (sets_seen [] e) codes ->
    incl (sets_through [] e) seen ->
    groups_whole e = true ->
    program_refines e (know_code_through sets codes seen e).

(* So what [read_slots_through] gives refines the unit, run on its
   own. *)
Definition read_slots_through_program_refines_statement : Prop :=
  ∀ e : expr, program_refines e (read_slots_through e).

(* The same of a unit that imports symbols, each run on its own from
   the same imports (FlambdaImportsSpec.unit_refines). *)
Definition read_slots_through_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (e : expr),
    unit_refines imports e (read_slots_through e).

(* The same of the unit followed by what may follow a unit, and by
   nothing else: the unit and the unit rewritten, run from one
   continuation environment of the observers of
   FlambdaObserversSpec.v at the value slots given and from the static
   environment of the imports. *)
Definition read_slots_through_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (slots : list value_slot)
    (conts : cont_env) (e : expr),
    observers slots conts ->
    refines (State e [] conts (imports_statics imports) empty_heap no_reads)
      (State (read_slots_through e) [] conts (imports_statics imports)
         empty_heap no_reads).

(* The names of the result. Where the code of the unit is closed
   (FlambdaBindersSpec.code_closed), so is that of the result, and the
   variables and the continuations that are free in the result are
   free in the unit: what is put in the place of a projection is a
   projection from a variable that an earlier [Let] of the same body
   reads and that has not been bound again since. *)
Definition read_slots_through_names_statement : Prop :=
  ∀ e : expr,
    code_closed e = true ->
    code_closed (read_slots_through e) = true
    ∧ incl (free_variables_of_expr (read_slots_through e))
        (free_variables_of_expr e)
    ∧ incl (free_continuations_of_expr (read_slots_through e))
        (free_continuations_of_expr e).

(* The rewrite commutes with renaming, which renames no symbol, no code
   id and no slot. *)
Definition read_slots_through_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    read_slots_through (rename_expr r e) = rename_expr r (read_slots_through e).

(* ======================================================================== *)
(* With what else is known of the closures of the code of a unit            *)
(* ======================================================================== *)

(* FlambdaClosureOfCodeSpec.know_own_closures and then
   [read_slots_through], of what it gives. A round of the validation
   applies this where it applied the first alone
   (FlambdaValidateSpec.normalise_lifted_round_own). *)
Definition know_own_closures_through (e : expr) : expr :=
  read_slots_through (know_own_closures e).

(* The laws of the two, composed: the result refines the unit, run on
   its own, from its imports, and followed by what may follow a unit;
   it has no free variable or continuation that the unit has not, and
   its code is closed where the unit's is; and the pass commutes with
   renaming. *)
Definition know_own_closures_through_program_refines_statement : Prop :=
  ∀ e : expr, program_refines e (know_own_closures_through e).

Definition know_own_closures_through_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (e : expr),
    unit_refines imports e (know_own_closures_through e).

Definition know_own_closures_through_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (slots : list value_slot)
    (conts : cont_env) (e : expr),
    observers slots conts ->
    refines (State e [] conts (imports_statics imports) empty_heap no_reads)
      (State (know_own_closures_through e) [] conts (imports_statics imports)
         empty_heap no_reads).

Definition know_own_closures_through_names_statement : Prop :=
  ∀ e : expr,
    code_closed e = true ->
    code_closed (know_own_closures_through e) = true
    ∧ incl (free_variables_of_expr (know_own_closures_through e))
        (free_variables_of_expr e)
    ∧ incl (free_continuations_of_expr (know_own_closures_through e))
        (free_continuations_of_expr e).

Definition know_own_closures_through_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    know_own_closures_through (rename_expr r e)
    = rename_expr r (know_own_closures_through e).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The rewrite computed on small units. Each unit binds the code
   [inner_code] of a closure that is captured and the code
   [outer_code] of a closure that captures it, whose body reads
   [slot_a] of its closure and then [slot_d] of what that gave; and
   then makes closures of the two. What differs from one example to
   the next is what the closures are given and where. The last
   examples have a third code, whose closure holds a closure of the
   second: there the closure read from is itself read from a value
   slot. *)
Section Examples.

Let u : variable := 1%positive.
Let t : variable := 2%positive.
Let w : variable := 3%positive.
Let p : variable := 4%positive.
Let my_closure : variable := 5%positive.
Let y : variable := 6%positive.
Let z : variable := 7%positive.
Let q : variable := 8%positive.
Let h : variable := 9%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 3%positive.
Let k : continuation := 4%positive.
Let inner_code : code_id := 1%positive.
Let outer_code : code_id := 2%positive.
Let maker_code : code_id := 3%positive.
Let holder_code : code_id := 4%positive.
Let slot_a : value_slot := 1%positive.
Let slot_b : value_slot := 2%positive.
Let slot_d : value_slot := 3%positive.
Let sym : symbol := 1%positive.
Let other_function_slot : function_slot := 2%positive.

Definition through_example_arity : arity := [Singleton Value].

Definition through_example_metadata : code_metadata :=
  make_code_metadata through_example_arity through_example_arity.

Definition through_example_seven : simple := Const (Tagged_immediate 7).

Definition through_example_eight : simple := Const (Tagged_immediate 8).

(* The code of the closure that is captured: it returns its
   argument. *)
Definition through_example_inner : code :=
  Code [p] my_closure ret exn through_example_metadata
    (Apply_cont (Apply_cont_expr ret [Var p] None)).

(* The projection of [slot_a] from the closure of the code that
   runs. *)
Definition through_example_first : named :=
  Project_value_slot_from unnamed_function_slot slot_a (Var my_closure).

(* The read through the closure that the variable [from] is bound
   to. *)
Definition through_example_read (from : variable) : named :=
  Project_value_slot_from unnamed_function_slot slot_d (Var from).

(* The read of the value slot of the closure itself. *)
Definition through_example_own : named :=
  Project_value_slot_from unnamed_function_slot slot_b (Var my_closure).

(* The code of the closure that captures: it binds [first] to
   [slot_a] of its closure and returns what [read] gives. *)
Definition through_example_outer (first : variable) (read : named) : code :=
  Code [p] my_closure ret exn through_example_metadata
    (Let first Value through_example_first
       (Let z Value read (Apply_cont (Apply_cont_expr ret [Var z] None)))).

(* A unit that binds the two pieces of code, the second as given, and
   goes on as [body]. *)
Definition through_example_unit (outer : code) (body : expr) : expr :=
  Let_static
    [Bind_code inner_code through_example_inner; Bind_code outer_code outer]
    body.

(* The unit whose second code reads [slot_d] through the closure in
   [slot_a], and the unit that reads [slot_b] of the closure itself
   in its place. *)
Definition through_example_before (body : expr) : expr :=
  through_example_unit
    (through_example_outer y (through_example_read y)) body.

Definition through_example_after (body : expr) : expr :=
  through_example_unit (through_example_outer y through_example_own) body.

(* A closure of the first code and a closure of the second, bound to
   [x], with the value slots given. *)
Definition through_example_make_inner (x : variable)
    (value_slots : list (value_slot * simple)) (body : expr) : expr :=
  Let x Value (Set_of_closures inner_code value_slots) body.

Definition through_example_make_outer (x : variable)
    (value_slots : list (value_slot * simple)) (body : expr) : expr :=
  Let x Value (Set_of_closures outer_code value_slots) body.

Definition through_example_done (x : variable) : expr :=
  Apply_cont (Apply_cont_expr done [Var x] None).

(* A closure [t] of the first code with [slot_d] given [in_d], and a
   closure [w] of the second with [slot_a] given [t] and [slot_b]
   given [in_b]. *)
Definition through_example_pair (in_d in_b : simple) (body : expr) : expr :=
  through_example_make_inner t [(slot_d, in_d)]
    (through_example_make_outer w [(slot_a, Var t); (slot_b, in_b)] body).

Definition through_example_let_seven (x : variable) (body : expr) : expr :=
  Let x Value (Simple through_example_seven) body.

(* The reads that are replaced. *)

(* The two value slots are given one variable. *)
Definition read_slots_through_reads_statement : Prop :=
  let body :=
    through_example_let_seven u
      (through_example_pair (Var u) (Var u) (through_example_done w))
  in
  read_slots_through (through_example_before body)
  = through_example_after body.

(* One constant. *)
Definition read_slots_through_constant_statement : Prop :=
  let body :=
    through_example_pair through_example_seven through_example_seven
      (through_example_done w)
  in
  read_slots_through (through_example_before body)
  = through_example_after body.

(* One symbol. *)
Definition read_slots_through_symbol_statement : Prop :=
  let body :=
    through_example_pair (Symbol sym) (Symbol sym) (through_example_done w)
  in
  read_slots_through (through_example_before body)
  = through_example_after body.

(* Two places make the closures, with a variable in one and a
   constant in the other: what is common is common at each place, not
   to the two. *)
Definition read_slots_through_two_places_statement : Prop :=
  let body :=
    through_example_let_seven u
      (through_example_pair (Var u) (Var u)
         (through_example_pair through_example_eight through_example_eight
            (through_example_done w)))
  in
  read_slots_through (through_example_before body)
  = through_example_after body.

(* The second closure is made in a handler that has no parameter, and
   the variable is bound again before the handler is applied: the
   handler runs under the variables of the place where it is defined,
   where the variable is the one that the first closure was given. *)
Definition read_slots_through_handler_statement : Prop :=
  let body :=
    through_example_let_seven u
      (through_example_make_inner t [(slot_d, Var u)]
         (Let_cont_non_recursive k
            (Handler [] (make_handler_metadata [])
               (through_example_make_outer w
                  [(slot_a, Var t); (slot_b, Var u)]
                  (through_example_done w)))
            (Let u Value (Simple through_example_eight)
               (Apply_cont (Apply_cont_expr k [] None)))))
  in
  read_slots_through (through_example_before body)
  = through_example_after body.

(* The two closures are made in a loop, of the parameter of the loop,
   which is another value each time round. *)
Definition read_slots_through_loop_statement : Prop :=
  let body :=
    Let_cont_recursive
      [(k,
        Handler [u] (make_handler_metadata [Value])
          (through_example_pair (Var u) (Var u)
             (Apply_cont (Apply_cont_expr k [Var w] None))))]
      (Apply_cont (Apply_cont_expr k [through_example_seven] None))
  in
  read_slots_through (through_example_before body)
  = through_example_after body.

(* The two closures are made in the body of code, of its parameter. *)
Definition read_slots_through_in_code_statement : Prop :=
  let unit (outer : code) : expr :=
    Let_static
      [Bind_code inner_code through_example_inner;
       Bind_code outer_code outer;
       Bind_code maker_code
         (Code [p] my_closure ret exn through_example_metadata
            (through_example_pair (Var p) (Var p)
               (Apply_cont (Apply_cont_expr ret [Var w] None))))]
      (Apply_cont (Apply_cont_expr done [] None))
  in
  read_slots_through
    (unit (through_example_outer y (through_example_read y)))
  = unit (through_example_outer y through_example_own).

(* The function slot that the projection replaced names is not looked
   at. Here the closure in [slot_a] is not at the function slot
   named: if this code is called with the closure that the unit
   makes, its second projection has no value, and the one put in its
   place has. This statement is of the rewrite only: the unit gives
   the closure to its exit without calling it, and nothing is said
   here of a run. [read_slots_through_other_function_slot_run_statement]
   is the run. *)
Definition read_slots_through_other_function_slot_statement : Prop :=
  let body :=
    through_example_let_seven u
      (through_example_pair (Var u) (Var u) (through_example_done w))
  in
  read_slots_through
    (through_example_unit
       (through_example_outer y
          (Project_value_slot_from other_function_slot slot_d (Var y)))
       body)
  = through_example_after body.

(* The reads that are left. In each of these the unit is what
   [read_slots_through] gives of it. *)

Definition through_example_left (body : expr) : Prop :=
  read_slots_through (through_example_before body)
  = through_example_before body.

(* The two value slots are given two simples, which here have one
   value when the unit runs: the test is of the simples as they are
   written. *)
Definition read_slots_through_other_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (through_example_pair (Var u) through_example_seven
          (through_example_done w))).

(* One is given a variable and the other a variable bound to it. *)
Definition read_slots_through_alias_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (Let q Value (Simple (Var u))
          (through_example_pair (Var u) (Var q) (through_example_done w)))).

(* Two places make the closures and the second gives two simples: one
   place that does not pass is enough. *)
Definition read_slots_through_one_place_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (through_example_pair (Var u) (Var u)
          (through_example_pair (Var u) through_example_eight
             (through_example_done w)))).

(* The variable is bound again between the two closures: the first
   holds seven and the second eight. *)
Definition read_slots_through_bound_between_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (through_example_make_inner t [(slot_d, Var u)]
          (Let u Value (Simple through_example_eight)
             (through_example_make_outer w
                [(slot_a, Var t); (slot_b, Var u)]
                (through_example_done w))))).

(* The [Let] that makes the first closure binds the variable that the
   closure is given: [slot_d] holds the old value of the variable, a
   number, and the second closure is given the new one, the first
   closure, in both its value slots. *)
Definition read_slots_through_own_binder_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (through_example_make_inner u [(slot_d, Var u)]
          (through_example_make_outer w
             [(slot_a, Var u); (slot_b, Var u)]
             (through_example_done w)))).

(* The second closure is made in a handler one of whose parameters
   has the name of the variable that the first closure was given: the
   first holds seven and the second eight. *)
Definition read_slots_through_parameter_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (through_example_make_inner t [(slot_d, Var u)]
          (Let_cont_non_recursive k
             (Handler [u] (make_handler_metadata [Value])
                (through_example_make_outer w
                   [(slot_a, Var t); (slot_b, Var u)]
                   (through_example_done w)))
             (Apply_cont (Apply_cont_expr k [through_example_eight] None))))).

(* The first closure comes round a loop as a parameter: it is not in
   sight where the second is made, whatever code it is of. The second
   time round it holds seven and the second closure is given eight. *)
Definition read_slots_through_carried_statement : Prop :=
  through_example_left
    (Let_cont_recursive
       [(k,
         Handler [t; u] (make_handler_metadata [Value; Value])
           (through_example_make_outer w [(slot_a, Var t); (slot_b, Var u)]
              (Apply_cont
                 (Apply_cont_expr k [Var t; through_example_eight] None))))]
       (through_example_let_seven u
          (through_example_make_inner t [(slot_d, Var u)]
             (Apply_cont (Apply_cont_expr k [Var t; Var u] None))))).

(* The first closure is given [slot_d] twice: it holds the first, a
   constant, and the second closure is given what the second was
   given. *)
Definition read_slots_through_twice_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (through_example_make_inner t
          [(slot_d, through_example_eight); (slot_d, Var u)]
          (through_example_make_outer w [(slot_a, Var t); (slot_b, Var u)]
             (through_example_done w)))).

(* It is given [slot_d] twice and the variable that the first was
   given is bound again: the first is forgotten, and the second is not
   taken for it. The closure holds seven in [slot_d], and the second
   closure is given eight. *)
Definition read_slots_through_twice_forgotten_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (Let q Value (Simple through_example_eight)
          (through_example_make_inner t [(slot_d, Var u); (slot_d, Var q)]
             (Let u Value (Simple through_example_eight)
                (through_example_make_outer w
                   [(slot_a, Var t); (slot_b, Var q)]
                   (through_example_done w)))))).

(* The first closure has no [slot_d]; and the second no [slot_b]. *)
Definition read_slots_through_no_slot_statement : Prop :=
  through_example_left
    (through_example_make_inner t []
       (through_example_make_outer w
          [(slot_a, Var t); (slot_b, through_example_seven)]
          (through_example_done w)))
  ∧ through_example_left
      (through_example_make_inner t [(slot_d, through_example_seven)]
         (through_example_make_outer w [(slot_a, Var t)]
            (through_example_done w))).

(* The second closure is given, in [slot_a], a variable bound to the
   projection of a function slot from the first: that variable is not
   in sight. *)
Definition read_slots_through_projected_statement : Prop :=
  through_example_left
    (through_example_let_seven u
       (through_example_make_inner t [(slot_d, Var u)]
          (Let q Value
             (Project_function_slot unnamed_function_slot
                unnamed_function_slot (Var t))
             (through_example_make_outer w
                [(slot_a, Var q); (slot_b, Var u)]
                (through_example_done w))))).

(* The first closure is static and the second is given its symbol:
   nothing is known of a symbol. The groups of the unit bind their
   static sets whole, so that is not why it is left. *)
Definition read_slots_through_lifted_statement : Prop :=
  let body :=
    Let_static
      [Bind_symbol sym
         (Static_const_set_of_closures inner_code
            [(slot_d, through_example_seven)])]
      (through_example_make_outer w
         [(slot_a, Symbol sym); (slot_b, through_example_seven)]
         (through_example_done w))
  in
  groups_whole (through_example_before body) = true
  ∧ through_example_left body.

(* In the body of the code, the [Let] of the first projection binds
   the variable that it reads: what it binds is the closure in
   [slot_a], and the closure of the code is no longer in reach. *)
Definition read_slots_through_reads_own_binder_statement : Prop :=
  let unit :=
    through_example_unit
      (through_example_outer my_closure (through_example_read my_closure))
      (through_example_let_seven u
         (through_example_pair (Var u) (Var u) (through_example_done w)))
  in
  read_slots_through unit = unit.

(* In the body of the code, the variable that the first projection
   reads is bound again before the second projection. *)
Definition read_slots_through_base_bound_statement : Prop :=
  let unit :=
    through_example_unit
      (Code [p] my_closure ret exn through_example_metadata
         (Let y Value through_example_first
            (Let my_closure Value (Simple (Var p))
               (Let z Value (through_example_read y)
                  (Apply_cont (Apply_cont_expr ret [Var z] None))))))
      (through_example_let_seven u
         (through_example_pair (Var u) (Var u) (through_example_done w)))
  in
  read_slots_through unit = unit.

(* The closure read from is itself read from a value slot. A third
   code, [holder_code]: its closure holds in [slot_a] a closure of the
   second code. Its body binds [q] to that closure, [y] to the closure
   that [q] holds in [slot_a], and returns what [read] gives. *)
Definition through_example_holder (read : named) : code :=
  Code [p] my_closure ret exn through_example_metadata
    (Let q Value through_example_first
       (Let y Value
          (Project_value_slot_from unnamed_function_slot slot_a (Var q))
          (Let z Value read
             (Apply_cont (Apply_cont_expr ret [Var z] None))))).

(* The read of [slot_b] of the closure that [q] is bound to. *)
Definition through_example_of_held : named :=
  Project_value_slot_from unnamed_function_slot slot_b (Var q).

(* A unit that binds the three pieces of code, the second reading
   [slot_b] of its own closure and the third as given, and goes on as
   [body]. *)
Definition through_example_held_unit (holder : code) (body : expr) : expr :=
  Let_static
    [Bind_code inner_code through_example_inner;
     Bind_code outer_code (through_example_outer y through_example_own);
     Bind_code holder_code holder]
    body.

(* A closure of the third code, bound to [x], with [slot_a] given. *)
Definition through_example_make_holder (x : variable) (in_a : simple)
    (body : expr) : expr :=
  Let x Value (Set_of_closures holder_code [(slot_a, in_a)]) body.

(* The third code reads the closure of the second from its own
   closure, the closure of the first through that, and [slot_d]
   through that. The one set with the third code gives its [slot_a]
   the closure of the second code, which is in sight; so [q] is known
   to be a closure of the second code, and the last read becomes the
   read of [slot_b] of [q]. It does not become a read of
   [my_closure], which has no such value slot. *)
Definition read_slots_through_held_statement : Prop :=
  let body :=
    through_example_let_seven u
      (through_example_pair (Var u) (Var u)
         (through_example_make_holder h (Var w) (through_example_done h)))
  in
  read_slots_through
    (through_example_held_unit
       (through_example_holder (through_example_read y)) body)
  = through_example_held_unit
      (through_example_holder through_example_of_held) body.

(* The third closure is given a variable bound to the closure of the
   second code, and not the variable that the [Let] of that closure
   bound: what its [slot_a] holds is not in sight, [q] is not known,
   and the read is left. *)
Definition read_slots_through_held_alias_statement : Prop :=
  let unit :=
    through_example_held_unit
      (through_example_holder (through_example_read y))
      (through_example_let_seven u
         (through_example_pair (Var u) (Var u)
            (Let q Value (Simple (Var w))
               (through_example_make_holder h (Var q)
                  (through_example_done h)))))
  in
  read_slots_through unit = unit.

(* Two places make closures of the third code, and the second gives
   [slot_a] the closure of the first code: what the value slot holds
   has not one code, [q] is not known, and the read is left. *)
Definition read_slots_through_held_two_codes_statement : Prop :=
  let unit :=
    through_example_held_unit
      (through_example_holder (through_example_read y))
      (through_example_let_seven u
         (through_example_pair (Var u) (Var u)
            (through_example_make_holder h (Var w)
               (through_example_make_holder h (Var t)
                  (through_example_done h)))))
  in
  read_slots_through unit = unit.

(* The closure of the second code is static and the third closure is
   given its symbol: nothing is known of what a value slot holds
   where a set gives it a symbol, so [q] is not known and the read is
   left. The groups of the unit bind their static sets whole, and the
   sets with the second code pass the test of the two value slots:
   neither is why it is left. *)
Definition read_slots_through_held_symbol_statement : Prop :=
  let unit :=
    through_example_held_unit
      (through_example_holder (through_example_read y))
      (through_example_make_inner t [(slot_d, through_example_seven)]
         (Let_static
            [Bind_symbol sym
               (Static_const_set_of_closures outer_code
                  [(slot_a, Var t); (slot_b, through_example_seven)])]
            (through_example_make_holder h (Symbol sym)
               (through_example_done h))))
  in
  groups_whole unit = true
  ∧ sets_through_give outer_code unnamed_function_slot slot_a slot_b slot_d
      (sets_through [] unit)
    = true
  ∧ read_slots_through unit = unit.

(* Two runs. The unit makes the two closures as the first example does
   and calls the second with a number; the code of the second returns
   what its read gives. *)
Definition through_example_called (outer : code) : expr :=
  through_example_unit outer
    (through_example_let_seven u
       (through_example_pair (Var u) (Var u)
          (Apply
             (Apply_expr (Var w) Indirect_unknown_arity
                [through_example_eight] done exn [] through_example_arity
                through_example_arity)))).

(* Where the read names the function slot that the closure in [slot_a]
   is at, the unit and what [read_slots_through] gives of it both end
   with seven. *)
Definition read_slots_through_run_statement : Prop :=
  ∀ e : expr,
    e
    ∈ [through_example_called
         (through_example_outer y (through_example_read y));
       read_slots_through
         (through_example_called
            (through_example_outer y (through_example_read y)))] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads)
      (Exits done [Value_const (Tagged_immediate 7)] empty_heap).

(* Where it names another, the unit goes wrong at that projection, and
   what [read_slots_through] gives of it ends with seven: the rewrite
   refines, and is no equivalence. *)
Definition read_slots_through_other_function_slot_run_statement : Prop :=
  let unit :=
    through_example_called
      (through_example_outer y
         (Project_value_slot_from other_function_slot slot_d (Var y)))
  in
  behaves (State unit [] [] empty_static_env empty_heap no_reads) Goes_wrong
  ∧ behaves
      (State (read_slots_through unit) [] [] empty_static_env empty_heap
         no_reads)
      (Exits done [Value_const (Tagged_immediate 7)] empty_heap).

(* A run of the unit whose third code reads through two closures: it
   makes the three closures and calls the third with a number. The
   unit and what [read_slots_through] gives of it both end with
   seven. *)
Definition read_slots_through_held_run_statement : Prop :=
  let unit :=
    through_example_held_unit
      (through_example_holder (through_example_read y))
      (through_example_let_seven u
         (through_example_pair (Var u) (Var u)
            (through_example_make_holder h (Var w)
               (Apply
                  (Apply_expr (Var h) Indirect_unknown_arity
                     [through_example_eight] done exn [] through_example_arity
                     through_example_arity)))))
  in
  ∀ e : expr,
    e ∈ [unit; read_slots_through unit] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads)
      (Exits done [Value_const (Tagged_immediate 7)] empty_heap).

End Examples.
