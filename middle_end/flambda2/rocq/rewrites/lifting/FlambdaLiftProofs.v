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

(* Proofs of the statements of FlambdaLiftSpec.v. Nothing here needs
   review.

   The term and the term lifted are run side by side. Where the first
   executes a [Let] that is lifted, the second executes the [Let_static]
   and then the [Let] of the symbol, two steps for one; everywhere else
   they step together. So the simulation is one in which each step of
   the first run is answered by one or more of the second
   (FlambdaSimulationProofs.stuttering_heaps_refines). The relation
   between the two states
   ([state_lifted]) is that the second expression is the first lifted,
   the values of the second are those of the first up to
   [value_lifted], and so are what the symbols are bound to
   ([statics_lifted]) and what the cells of the two heaps hold
   (FlambdaHeapSpec.heap_alike). The machine cannot tell such values
   apart: every observation answers alike ([closure_of_lifted], ...),
   and so does every primitive, on two heaps alike
   (FlambdaPrimitiveProofs.eval_prim_related,
   FlambdaHeapProofs.exec_prim_related).

   What keeps a new static binding from failing where the [Let] did not
   is that the second run is that of a well-formed unit, which never
   binds a name twice (FlambdaWellFormedProofs.well_formed_never_faults).

   A call of code in which [my_closure] is bound to a symbol is also two
   steps for one, and the closure called must then be that symbol. A
   call may call the callee of an [Apply], a value that another call
   returned (what is left of an over-application calls it) or the
   closure that a partial application keeps. So the simulation has, of
   the second run, the invariant of FlambdaMyClosureProofs.v
   ([known_state_ok]): no value of it is a closure of that code but the
   symbol. A call is gone through by its shape ([shape_lifted]): the
   two runs make a full call, an over-application or a partial
   application together.

   That the values at the exit are equal, and not only lifted, is from
   an invariant of the first run alone ([plain_state]): the
   continuations in scope never stand for a way out ([cont_closed]),
   so a value that is not a constant or a symbol, a partial application
   included, is given to a handler; the variables known hold constants
   and symbols ([vars_plain]); and each continuation bound may be
   given anything at the positions that the expression spoils of it,
   and constants and symbols at the others ([cont_plain],
   [conts_fit]). So a handler is applied to a constant or a symbol
   where its parameter is known. What a handler of a call is given is
   what the code called gives its exception continuation, and code
   that passes the test spoils nothing of a continuation that is not
   of its scope ([knowing_spoils_none]), which is its return
   continuation alone. Where code gives its two continuations one
   name, the name is of its scope and stands for the return
   continuation: the call binds that one first, and its entry hides
   that of the exception continuation ([call_code_plain]).

   That the heaps at the exit are equal, and not only alike, is from
   the guard of FlambdaLiftSpec.lift_unit: a unit with no primitive of
   the heap exits with the empty heap
   (FlambdaHeapFreeProofs.free_of_heap_exits_empty), and a heap alike
   to the empty heap is empty ([heap_alike_empty]).

   The two runs may start from a continuation environment of what may
   follow a unit (FlambdaObserversSpec.observers), the empty one for
   the statements of FlambdaLiftSpec.v. What may follow a unit has no
   [Let] that is lifted and no code, so it is related to itself
   ([observer_lifted]), and it makes no closure of the code of a
   function ([observer_closure_ok]). Without the test of the exits the
   values of two exits that answer each other are lifted, so they have
   the same constants and the same locations ([lifted_same_scalars]),
   and their heaps are alike, so they bind the same locations:
   that is the statement of FlambdaLiftObservedSpec.v
   ([unit_lifted_observed]). With no [Let] lifted and no code named,
   the simulation is also that of one term run on values lifted, under
   static environments with the same code, and it serves a pass after
   whose two exits what follows is the same term
   ([observer_applied_lifted]). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveProofs
  FlambdaHeapSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaSimulationProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaClosuresSpec FlambdaClosuresProofs
  FlambdaUnusedStaticsSpec FlambdaMyClosureSpec FlambdaMyClosureProofs
  FlambdaImportsSpec FlambdaImportsProofs FlambdaStaticsInScopeSpec
  FlambdaRunUnderSpec FlambdaObserversSpec FlambdaObserversProofs
  FlambdaUnitFromProofs FlambdaHeapFreeSpec FlambdaHeapFreeProofs
  FlambdaLiftSpec FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* Lists and maps                                                           *)
(* ======================================================================== *)

(* Two maps with the same keys in the same order bind the same names. *)
Lemma lookup_same_keys {A B : Type} (name : positive)
    (l : list (positive * A)) :
  ∀ l' : list (positive * B),
    map fst l = map fst l' ->
    ∀ a : A, lookup name l = Some a -> ∃ b : B, lookup name l' = Some b.
Proof.
  induction l as [ | [k a0] l IH]; intros [ | [k' b0] l']; simpl;
    try discriminate.
  intros [= <- Hrest] a.
  destruct (Pos.eqb name k); [ eauto | apply IH; exact Hrest ].
Qed.

(* The first elements of two lists related pairwise are related, and so
   are the rest. *)
Lemma Forall2_firstn_skipn {A B : Type} (R : A -> B -> Prop) (n : nat) :
  ∀ (l : list A) (l' : list B),
    Forall2 R l l' ->
    Forall2 R (firstn n l) (firstn n l') ∧ Forall2 R (skipn n l) (skipn n l').
Proof.
  induction n as [ | n IH]; intros l l' H; simpl;
    [ split; [ constructor | exact H ] | ].
  destruct H as [ | a b l l' Hab H]; [ split; constructor | ].
  destruct (IH l l' H) as [Hfirst Hrest].
  split; [ constructor; assumption | exact Hrest ].
Qed.

(* ======================================================================== *)
(* Lifted values                                                            *)
(* ======================================================================== *)

Ltac lifted_cases H :=
  destruct H
    as [ c | s | c | tag fields fields' Hfields
       | items items' Hitems
       | code slots slots' Hslots | fslot decls slots slots' Hslots
       | pcallee pcallee' pargs pargs' parity Hpcallee Hpargs
       | c s Hlookup | tag fields fields' s Hlookup Hfields
       | items items' s Hlookup Hitems
       | fslot decls slots slots' s syms Hsym Hsyms Hbound Hslots
       | code slots slots' s Hlookup Hslots | loc ].

(* A value lifted under a static environment is lifted under one that
   extends it: a symbol stays bound to what it was bound to. *)
Fixpoint value_lifted_extend (statics' statics'' : static_env)
    (Hextend : statics_extend statics' statics'') (v v' : value)
    (H : value_lifted statics' v v') {struct H} :
  value_lifted statics'' v v'.
Proof.
  lifted_cases H.
  - apply Lifted_const.
  - apply Lifted_symbol.
  - apply Lifted_boxed_number.
  - apply Lifted_block.
    revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_lifted_extend _ _ Hextend _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply Lifted_array.
    revert items items' Hitems. fix IH 3.
    intros items items' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_lifted_extend _ _ Hextend _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply Lifted_closure.
    revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; constructor;
      [ split;
        [ exact Hkey | exact (value_lifted_extend _ _ Hextend _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply Lifted_closure_of_set.
    revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; constructor;
      [ split;
        [ exact Hkey | exact (value_lifted_extend _ _ Hextend _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply Lifted_partial_application;
      [ exact (value_lifted_extend _ _ Hextend _ _ Hpcallee) | ].
    clear Hpcallee. revert pargs pargs' Hpargs. fix IH 3.
    intros args args' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_lifted_extend _ _ Hextend _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply Lifted_to_boxed_number. exact (proj2 Hextend s _ Hlookup).
  - apply (Lifted_to_block statics'' tag fields fields' s);
      [ exact (proj2 Hextend s _ Hlookup) | ].
    clear Hlookup. revert fields fields' Hfields. fix IH 3.
    intros fields fields' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_lifted_extend _ _ Hextend _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply (Lifted_to_array statics'' items items' s);
      [ exact (proj2 Hextend s _ Hlookup) | ].
    clear Hlookup. revert items items' Hitems. fix IH 3.
    intros items items' [ | w w' rest rest' Hhead Htail]; constructor;
      [ exact (value_lifted_extend _ _ Hextend _ _ Hhead)
      | exact (IH _ _ Htail) ].
  - apply (Lifted_to_closure statics'' fslot decls slots slots' s syms Hsym
             Hsyms);
      [ exact (static_set_bound_extend _ _ _ _ _ Hbound Hextend) | ].
    clear Hbound. revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; constructor;
      [ split;
        [ exact Hkey | exact (value_lifted_extend _ _ Hextend _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply (Lifted_to_unnamed_closure statics'' code slots slots' s);
      [ exact (proj2 Hextend s _ Hlookup) | ].
    clear Hlookup. revert slots slots' Hslots. fix IH 3.
    intros slots slots' [ | w w' rest rest' [Hkey Hhead] Htail]; constructor;
      [ split;
        [ exact Hkey | exact (value_lifted_extend _ _ Hextend _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply Lifted_location.
Qed.

Lemma values_lifted_extend (statics' statics'' : static_env)
    (vs vs' : list value) :
  statics_extend statics' statics'' ->
  Forall2 (value_lifted statics') vs vs' ->
  Forall2 (value_lifted statics'') vs vs'.
Proof.
  intros Hextend. induction 1 as [ | v v' vs vs' Hv _ IH]; constructor;
    [ exact (value_lifted_extend _ _ Hextend _ _ Hv) | exact IH ].
Qed.

Lemma slots_lifted_extend (statics' statics'' : static_env)
    (slots slots' : list (value_slot * value)) :
  statics_extend statics' statics'' ->
  Forall2 (slot_lifted (value_lifted statics')) slots slots' ->
  Forall2 (slot_lifted (value_lifted statics'')) slots slots'.
Proof.
  intros Hextend.
  induction 1 as [ | slot slot' slots slots' [Hkey Hv] _ IH]; constructor;
    [ split; [ exact Hkey | exact (value_lifted_extend _ _ Hextend _ _ Hv) ]
    | exact IH ].
Qed.

(* A constant or a symbol is lifted to itself only. *)
Lemma plain_value_lifted (statics' : static_env) (v v' : value) :
  plain_value v -> value_lifted statics' v v' -> v' = v.
Proof. intros Hplain H. lifted_cases H; simpl in Hplain; tauto. Qed.

Lemma plain_values_lifted (statics' : static_env) (vs vs' : list value) :
  (∀ v : value, v ∈ vs -> plain_value v) ->
  Forall2 (value_lifted statics') vs vs' ->
  vs' = vs.
Proof.
  intros Hplain. induction 1 as [ | v v' vs vs' Hv _ IH]; [ reflexivity | ].
  rewrite (plain_value_lifted statics' v v' (Hplain v (or_introl eq_refl)) Hv),
    IH;
    [ reflexivity | ].
  intros w Hw. apply Hplain. right. exact Hw.
Qed.

(* Values lifted have the same constants and the same locations at the
   same places: a constant is lifted to itself only and from itself
   only, and so is a location. *)
Lemma lifted_same_scalars (statics' : static_env) (vs vs' : list value) :
  Forall2 (value_lifted statics') vs vs' -> same_scalars vs vs'.
Proof.
  unfold same_scalars.
  induction 1 as [ | v v' vs vs' Hv _ IH]; constructor; [ | exact IH ].
  intros [[[c ->] | [l ->]] | [[c ->] | [l ->]]];
    inversion Hv; reflexivity.
Qed.

(* A value lifted has the kind of the value: a symbol stands for what is
   allocated, of kind [Value]. *)
Lemma kind_of_value_lifted (statics' : static_env) (v v' : value) :
  value_lifted statics' v v' -> kind_of_value v' = kind_of_value v.
Proof. intros H. lifted_cases H; reflexivity. Qed.

Lemma kinds_of_values_lifted (statics' : static_env) (vs vs' : list value) :
  Forall2 (value_lifted statics') vs vs' ->
  map kind_of_value vs' = map kind_of_value vs.
Proof.
  induction 1 as [ | v v' vs vs' Hv _ IH]; simpl; [ reflexivity | ].
  rewrite (kind_of_value_lifted statics' v v' Hv), IH. reflexivity.
Qed.

(* A partial application is lifted to a partial application, and
   nothing else is. *)
Lemma partial_application_lifted (statics' : static_env) (v v' : value) :
  value_lifted statics' v v' ->
  match partial_application_of v with
  | Some (closure, applied, applied_arity) =>
    ∃ (closure' : value) (applied' : list value),
      v' = Value_partial_application closure' applied' applied_arity
      ∧ value_lifted statics' closure closure'
      ∧ Forall2 (value_lifted statics') applied applied'
  | None => partial_application_of v' = None
  end.
Proof.
  intros H. lifted_cases H; simpl; try reflexivity.
  exists pcallee', pargs'. auto.
Qed.

(* The value of a slot of lifted value slots is lifted. *)
Lemma lookup_slots_lifted (statics' : static_env) (slot : value_slot)
    (slots slots' : list (value_slot * value)) (v : value) :
  Forall2 (slot_lifted (value_lifted statics')) slots slots' ->
  lookup slot slots = Some v ->
  ∃ v' : value, lookup slot slots' = Some v' ∧ value_lifted statics' v v'.
Proof.
  induction 1 as [ | [k w] [k' w'] slots slots' [Hkey Hw] _ IH]; simpl;
    [ discriminate | ].
  simpl in Hkey, Hw. subst k'.
  destruct (Pos.eqb slot k); [ intros [= <-]; eauto | exact IH ].
Qed.

(* ======================================================================== *)
(* Lifted static environments                                               *)
(* ======================================================================== *)

(* What a symbol is bound to in the first run and in the second: the
   same, with lifted parts. *)
Inductive allocated_lifted (statics' : static_env)
    : statically_allocated_value -> statically_allocated_value -> Prop :=
  | Lifted_static_closure (code : code_id)
      (slots slots' : list (value_slot * value)) :
    Forall2 (slot_lifted (value_lifted statics')) slots slots' ->
    allocated_lifted statics' (Static_closure code slots)
      (Static_closure code slots')
  | Lifted_static_closure_of_set (fslot : function_slot)
      (syms : list (function_slot * symbol))
      (decls : list (function_slot * code_id))
      (slots slots' : list (value_slot * value)) :
    Forall2 (slot_lifted (value_lifted statics')) slots slots' ->
    allocated_lifted statics' (Static_closure_of_set fslot syms decls slots)
      (Static_closure_of_set fslot syms decls slots')
  | Lifted_static_block (tag : Z) (fields fields' : list value) :
    Forall2 (value_lifted statics') fields fields' ->
    allocated_lifted statics' (Static_block tag fields)
      (Static_block tag fields')
  | Lifted_static_boxed_number (v v' : value) :
    value_lifted statics' v v' ->
    allocated_lifted statics' (Static_boxed_number v) (Static_boxed_number v')
  | Lifted_static_string (bytes : list Z) :
    allocated_lifted statics' (Static_string bytes) (Static_string bytes)
  | Lifted_static_imported (c : import_class) :
    allocated_lifted statics' (Static_imported c) (Static_imported c)
  | Lifted_static_immutable_array (elements elements' : list value) :
    Forall2 (value_lifted statics') elements elements' ->
    allocated_lifted statics' (Static_immutable_array elements)
      (Static_immutable_array elements').

Ltac allocated_cases H :=
  destruct H
    as [ acode aslots aslots' Haslots
       | afslot asyms adecls aslots aslots' Haslots
       | atag afields afields' Hafields | av av' Hav | abytes |
       | aelements aelements' Haelements ].

Lemma allocated_lifted_extend (statics' statics'' : static_env)
    (allocated allocated' : statically_allocated_value) :
  statics_extend statics' statics'' ->
  allocated_lifted statics' allocated allocated' ->
  allocated_lifted statics'' allocated allocated'.
Proof.
  intros Hextend H. allocated_cases H; constructor;
    eauto using slots_lifted_extend, values_lifted_extend,
      value_lifted_extend.
Qed.

(* The second static environment binds every symbol that the first
   binds to the same thing, lifted. It may bind more symbols: those of
   the [Let]s lifted. *)
Definition symbols_lifted (statics statics' : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    ∃ allocated' : statically_allocated_value,
      lookup s (static_values statics') = Some allocated'
      ∧ allocated_lifted statics' allocated allocated'.

(* Every value is lifted to itself, and so is what a symbol is bound
   to: a static environment is lifted to itself, whatever it binds. *)
Fixpoint value_lifted_refl (statics' : static_env) (v : value) {struct v} :
  value_lifted statics' v v.
Proof.
  destruct v
    as [ c | code slots | fslot decls slots | s | tag fields | elements | c | l
       | callee args args_arity ].
  - apply Lifted_const.
  - apply Lifted_closure. revert slots. fix IH 1.
    intros [ | [slot w] slots]; constructor;
      [ split; [ reflexivity | exact (value_lifted_refl statics' w) ]
      | exact (IH slots) ].
  - apply Lifted_closure_of_set. revert slots. fix IH 1.
    intros [ | [slot w] slots]; constructor;
      [ split; [ reflexivity | exact (value_lifted_refl statics' w) ]
      | exact (IH slots) ].
  - apply Lifted_symbol.
  - apply Lifted_block. revert fields. fix IH 1.
    intros [ | w fields]; constructor;
      [ exact (value_lifted_refl statics' w) | exact (IH fields) ].
  - apply Lifted_array. revert elements. fix IH 1.
    intros [ | w elements]; constructor;
      [ exact (value_lifted_refl statics' w) | exact (IH elements) ].
  - apply Lifted_boxed_number.
  - apply Lifted_location.
  - apply Lifted_partial_application;
      [ exact (value_lifted_refl statics' callee) | ].
    revert args. fix IH 1.
    intros [ | w args]; constructor;
      [ exact (value_lifted_refl statics' w) | exact (IH args) ].
Qed.

Lemma values_lifted_refl (statics' : static_env) (vs : list value) :
  Forall2 (value_lifted statics') vs vs.
Proof.
  induction vs as [ | v vs IH]; constructor;
    [ apply value_lifted_refl | exact IH ].
Qed.

Lemma slots_lifted_refl (statics' : static_env)
    (slots : list (value_slot * value)) :
  Forall2 (slot_lifted (value_lifted statics')) slots slots.
Proof.
  induction slots as [ | [slot v] slots IH]; constructor;
    [ split; [ reflexivity | apply value_lifted_refl ] | exact IH ].
Qed.

Lemma allocated_lifted_refl (statics' : static_env)
    (allocated : statically_allocated_value) :
  allocated_lifted statics' allocated allocated.
Proof.
  destruct allocated; constructor;
    auto using value_lifted_refl, values_lifted_refl, slots_lifted_refl.
Qed.

Lemma symbols_lifted_refl (statics : static_env) :
  symbols_lifted statics statics.
Proof.
  intros s allocated Hs. exists allocated.
  split; [ exact Hs | apply allocated_lifted_refl ].
Qed.

(* Two heaps alike up to [value_lifted]: the cells of the second run
   hold the values of those of the first, lifted. A heap is alike to
   itself, and two heaps alike stay so under a static environment that
   extends the second. *)
Lemma heap_alike_impl (R R' : value -> value -> Prop) (h h' : heap) :
  (∀ v v' : value, R v v' -> R' v v') ->
  heap_alike R h h' ->
  heap_alike R' h h'.
Proof.
  intros Himpl.
  induction 1 as [ | [l c] [l' c'] h h' [Hkey Hcell] _ IH]; constructor;
    [ | exact IH ].
  split; [ exact Hkey | ]. simpl in Hcell |- *.
  destruct c as [mut tag fields | kind elements],
    c' as [mut' tag' fields' | kind' elements'];
    try contradiction.
  - destruct Hcell as (Hmut & Htag & Hfields).
    split; [ exact Hmut | split; [ exact Htag | ] ].
    exact (Forall2_impl _ Himpl Hfields).
  - destruct Hcell as (Hkind & Helements).
    split; [ exact Hkind | exact (Forall2_impl _ Himpl Helements) ].
Qed.

Lemma heap_lifted_extend (statics' statics'' : static_env) (h h' : heap) :
  statics_extend statics' statics'' ->
  heap_alike (value_lifted statics') h h' ->
  heap_alike (value_lifted statics'') h h'.
Proof.
  intros Hextend. apply heap_alike_impl. intros v v'.
  exact (value_lifted_extend _ _ Hextend v v').
Qed.

Lemma heap_lifted_refl (statics' : static_env) (h : heap) :
  heap_alike (value_lifted statics') h h.
Proof.
  induction h as [ | [l c] h IH]; constructor; [ | exact IH ].
  split; [ reflexivity | ].
  destruct c as [mut tag fields | kind elements]; simpl;
    auto using values_lifted_refl.
Qed.

(* A heap alike to the empty heap is empty. *)
Lemma heap_alike_empty (R : value -> value -> Prop) (h' : heap) :
  heap_alike R empty_heap h' -> h' = empty_heap.
Proof. unfold heap_alike, empty_heap. intros H. inversion H. reflexivity. Qed.

(* A location is lifted to itself, and to and from nothing else. *)
Lemma value_lifted_locations (statics' : static_env) :
  relates_locations (value_lifted statics').
Proof.
  split; [ intros l; apply Lifted_location | split ].
  - intros l v' H. inversion H. reflexivity.
  - intros v l H. inversion H. reflexivity.
Qed.

(* With no code id named, [my_closure] is bound nowhere: the term is
   left as it is. *)
Fixpoint bind_my_closures_nothing (e : expr) {struct e} :
  bind_my_closures [] e = e
with bind_my_closures_in_handler_nothing (h : continuation_handler)
    {struct h} :
  bind_my_closures_in_handler [] h = h
with bind_my_closures_in_binding_nothing (binding : static_binding)
    {struct binding} :
  bind_my_closures_in_binding [] binding = binding
with bind_my_closure_nothing (id : code_id) (c : code) {struct c} :
  bind_my_closure [] id c = c.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite bind_my_closures_nothing. reflexivity.
    + rewrite bind_my_closures_nothing. f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite bind_my_closures_in_binding_nothing, IH. reflexivity.
    + rewrite bind_my_closures_in_handler_nothing, bind_my_closures_nothing.
      reflexivity.
    + rewrite bind_my_closures_nothing. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite bind_my_closures_in_handler_nothing, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite bind_my_closures_nothing. reflexivity.
  - destruct binding as [id c | id | sym const]; simpl; try reflexivity.
    rewrite bind_my_closure_nothing. reflexivity.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    rewrite bind_my_closures_nothing. reflexivity.
Qed.

(* What may follow a unit names no code id, so it makes no closure of
   the code of a function: the check of FlambdaMyClosureSpec.v passes of
   it, for whatever code id and symbol. *)
Lemma named_in_scope_makes_no_closure (id : code_id) (syms : list symbol)
    (d : named) :
  named_statics_in_scope syms [] d = true ->
  named_makes_no_closure_of id d = true.
Proof.
  destruct d as [ | | code slots | slot decls slots | | | ]; simpl;
    try reflexivity.
  - discriminate.
  - intros H. apply andb_true_iff in H as [Hdecls _].
    destruct (map snd decls) as [ | code codes];
      [ reflexivity | discriminate Hdecls ].
Qed.

Fixpoint observer_only_closure_of (id : code_id) (s : symbol) (e : expr)
    {struct e} :
  binds_no_static e = true ->
  expr_statics_in_scope [] [] e = true ->
  only_closure_of id s e = true.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _ _; reflexivity).
  - intros Hno H. apply andb_true_iff in H as [Hd Hbody].
    rewrite (named_in_scope_makes_no_closure id [] d Hd).
    exact (observer_only_closure_of id s body Hno Hbody).
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros Hno H.
    apply andb_true_iff in Hno as [Hno_h Hno_b].
    apply andb_true_iff in H as [Hh Hb].
    rewrite (no_static_names symbols_of_binding body Hno_b),
      (no_static_names declared_code_ids_of_binding body Hno_b)
      in Hh.
    simpl in Hh.
    rewrite (observer_only_closure_of id s handler Hno_h Hh),
      (observer_only_closure_of id s body Hno_b Hb).
    reflexivity.
  - intros Hno H. apply andb_true_iff in Hno as [Hno_hs Hno_b].
    apply andb_true_iff in H as [Hhs Hb].
    rewrite (observer_only_closure_of id s body Hno_b Hb), andb_true_r.
    rewrite (no_static_names symbols_of_binding body Hno_b),
      (no_static_names declared_code_ids_of_binding body Hno_b)
      in Hhs.
    simpl in Hhs.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hno_hs, Hhs.
    apply andb_true_iff in Hno_hs as [Hno_h Hno_rest].
    apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (observer_only_closure_of id s handler Hno_h Hh).
    exact (IH Hno_rest Hrest).
Qed.

Lemma observer_closure_ok (id : code_id) (s : symbol)
    (slots : list value_slot) (c : cont_value) :
  observer slots c -> cont_ok id s c.
Proof.
  destruct c
    as [ [params metadata body] vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction.
  - destruct vars; [ | contradiction ].
    destruct conts; [ | contradiction ].
    destruct recursive; [ | contradiction ].
    intros (Hno & Hscope & _).
    apply Closure_ok;
      [ exact (observer_only_closure_of id s body Hno Hscope)
      | intros x v [=] | constructor | reflexivity ].
  - intros _. apply Outer_ok.
Qed.

Lemma observers_closures_ok (id : code_id) (s : symbol)
    (slots : list value_slot) (conts : cont_env) :
  observers slots conts -> conts_ok id s conts.
Proof.
  unfold observers, conts_ok. intros H.
  induction H as [ | [k c] conts Hc _ IH]; constructor;
    [ exact (observer_closure_ok id s slots c Hc) | exact IH ].
Qed.

Section Known.

(* The code ids whose code has [my_closure] bound to a symbol
   (FlambdaMyClosureSpec.bind_my_closures). *)
Variable known : list (code_id * symbol).

(* The second static environment has the code of the first, with
   [my_closure] bound where [known] says, and its symbols lifted. *)
Definition statics_lifted (statics statics' : static_env) : Prop :=
  (∀ id : code_id,
    lookup id (static_code statics')
    = option_map (bind_my_closure known id) (lookup id (static_code statics)))
  ∧ symbols_lifted statics statics'.

(* The second static environment may grow by symbols alone. *)
Lemma statics_lifted_extend (statics statics' statics'' : static_env) :
  statics_lifted statics statics' ->
  statics_extend statics' statics'' ->
  static_code statics'' = static_code statics' ->
  statics_lifted statics statics''.
Proof.
  intros [Hcode Hvalues] Hextend Hcode'.
  split; [ intros id; rewrite Hcode'; apply Hcode | ].
  intros s allocated Hs.
  destruct (Hvalues s allocated Hs) as (allocated' & Hs' & Hallocated).
  exists allocated'.
  split; [ exact (proj2 Hextend s _ Hs') | ].
  exact (allocated_lifted_extend _ _ _ _ Hextend Hallocated).
Qed.

(* ======================================================================== *)
(* The observations answer alike of a value and of the value lifted         *)
(* ======================================================================== *)

Section Observations.

Variables statics statics' : static_env.
Hypothesis Hsymbols : symbols_lifted statics statics'.

Lemma block_of_lifted (v v' : value) (block_tag : Z)
    (block_fields : list value) :
  value_lifted statics' v v' ->
  block_of statics v = Some (block_tag, block_fields) ->
  ∃ block_fields' : list value,
    block_of statics' v' = Some (block_tag, block_fields')
    ∧ Forall2 (value_lifted statics') block_fields block_fields'.
Proof.
  intros Hv. lifted_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_cases Hallocated; try discriminate.
    intros [= <- <-]. eauto.
  - intros [= <- <-]. eauto.
  - intros [= <- <-]. rewrite Hlookup. eauto.
Qed.

Lemma boxed_number_of_lifted (v v' : value) (number : const) :
  value_lifted statics' v v' ->
  boxed_number_of statics v = Some number ->
  boxed_number_of statics' v' = Some number.
Proof.
  intros Hv. lifted_cases Hv; simpl; try discriminate;
    try exact (fun H => H).
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_cases Hallocated; try discriminate.
    destruct av; try discriminate. inversion Hav. exact (fun H => H).
  - rewrite Hlookup. exact (fun H => H).
Qed.

Lemma string_of_lifted (v v' : value) (bytes : list Z) :
  value_lifted statics' v v' ->
  string_of statics v = Some bytes ->
  string_of statics' v' = Some bytes.
Proof.
  intros Hv. lifted_cases Hv; simpl; try discriminate.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
    [ | discriminate ].
  destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
  allocated_cases Hallocated; try discriminate. exact (fun H => H).
Qed.

Lemma array_of_lifted (v v' : value) (elements : list value) :
  value_lifted statics' v v' ->
  array_of statics v = Some elements ->
  ∃ elements' : list value,
    array_of statics' v' = Some elements'
    ∧ Forall2 (value_lifted statics') elements elements'.
Proof.
  intros Hv. lifted_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_cases Hallocated; try discriminate.
    intros [= <-]. eauto.
  - intros [= <-]. eauto.
  - intros [= <-]. rewrite Hlookup. eauto.
Qed.

Lemma closure_in_set_lifted (v v' : value) (at_slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  value_lifted statics' v v' ->
  closure_in_set statics v = Some (at_slot, function_decls, value_slots) ->
  ∃ value_slots' : list (value_slot * value),
    closure_in_set statics' v' = Some (at_slot, function_decls, value_slots')
    ∧ Forall2 (slot_lifted (value_lifted statics')) value_slots value_slots'.
Proof.
  intros Hv. lifted_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_cases Hallocated; try discriminate.
    + intros [= <- <- <-]. eauto.
    + destruct (lookup afslot adecls); [ | discriminate ]. simpl.
      intros [= <- <- <-]. eauto.
  - intros [= <- <- <-]. eauto.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <- <-]. eauto.
  - destruct (lookup fslot decls) as [code0 | ] eqn:Hcode; [ | discriminate ].
    simpl. intros [= <- <- <-].
    rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)), Hcode. simpl. eauto.
  - intros [= <- <- <-]. rewrite Hlookup. eauto.
Qed.

Lemma closure_of_lifted (v v' : value) (called : code_id)
    (value_slots : list (value_slot * value)) :
  value_lifted statics' v v' ->
  closure_of statics v = Some (called, value_slots) ->
  ∃ value_slots' : list (value_slot * value),
    closure_of statics' v' = Some (called, value_slots')
    ∧ Forall2 (slot_lifted (value_lifted statics')) value_slots value_slots'.
Proof.
  intros Hv. lifted_cases Hv; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_cases Hallocated; try discriminate.
    + intros [= <- <-]. eauto.
    + destruct (lookup afslot adecls); [ | discriminate ]. simpl.
      intros [= <- <-]. eauto.
  - intros [= <- <-]. eauto.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    intros [= <- <-]. eauto.
  - destruct (lookup fslot decls) as [code0 | ] eqn:Hcode; [ | discriminate ].
    simpl. intros [= <- <-].
    rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)), Hcode. simpl. eauto.
  - intros [= <- <-]. rewrite Hlookup. eauto.
Qed.

Lemma project_value_slot_lifted (project_from : function_slot)
    (slot : value_slot) (v v' w : value) :
  value_lifted statics' v v' ->
  project_value_slot statics project_from slot v = Some w ->
  ∃ w' : value,
    project_value_slot statics' project_from slot v' = Some w'
    ∧ value_lifted statics' w w'.
Proof.
  intros Hv. unfold project_value_slot.
  destruct (closure_in_set statics v) as [[[at_slot decls] slots] | ] eqn:Hc;
    [ | discriminate ].
  destruct (closure_in_set_lifted v v' at_slot decls slots Hv Hc)
    as (slots' & -> & Hslots).
  destruct (Pos.eqb at_slot project_from); [ | discriminate ].
  exact (lookup_slots_lifted statics' slot slots slots' w Hslots).
Qed.

Lemma project_function_slot_lifted (move_from move_to : function_slot)
    (v v' w : value) :
  value_lifted statics' v v' ->
  project_function_slot statics move_from move_to v = Some w ->
  ∃ w' : value,
    project_function_slot statics' move_from move_to v' = Some w'
    ∧ value_lifted statics' w w'.
Proof.
  intros Hv. unfold project_function_slot.
  lifted_cases Hv; try discriminate.
  - (* A symbol: bound alike in both runs. *)
    destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct (Hsymbols s allocated Hs) as (allocated' & -> & Hallocated).
    allocated_cases Hallocated; try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros [= <-]. eexists. split; [ reflexivity | apply Lifted_symbol ].
    + destruct (Pos.eqb afslot move_from); [ | discriminate ].
      destruct (lookup afslot adecls); [ | discriminate ].
      destruct (lookup move_to adecls); [ | discriminate ]. simpl.
      destruct (lookup move_to asyms) as [s' | ]; [ | discriminate ]. simpl.
      intros [= <-]. eexists. split; [ reflexivity | apply Lifted_symbol ].
  - (* A closure of a set of one function. *)
    destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_closure, Hslots.
  - (* A closure of a set, made by a [Let] in both runs. *)
    destruct (Pos.eqb fslot move_from); [ | discriminate ].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ]. simpl.
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_closure_of_set, Hslots.
  - (* A closure of a set, lifted: the second run has the symbol of the
       slot asked for. *)
    rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)).
    destruct (Pos.eqb fslot move_from); [ | discriminate ].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (lookup move_to decls) as [code' | ] eqn:Hto; [ | discriminate ].
    simpl.
    destruct (lookup_same_keys move_to decls syms (eq_sym Hsyms) code' Hto)
      as [s' Hs'].
    rewrite Hs'. simpl. intros [= <-]. eexists. split; [ reflexivity | ].
    exact (Lifted_to_closure statics' move_to decls slots slots' s' syms Hs'
             Hsyms Hbound Hslots).
  - (* A closure of a set of one function, lifted. *)
    rewrite Hlookup.
    destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. eexists. split; [ reflexivity | ].
    exact (Lifted_to_unnamed_closure statics' code slots slots' s Hlookup
             Hslots).
Qed.

(* The relation is one that the observations cannot tell from equality,
   so no primitive can. *)
Lemma value_lifted_respects :
  respects_observations (value_lifted statics') statics statics'.
Proof.
  split; [ | split ].
  - split.
    + intros c v' Hv. inversion Hv. reflexivity.
    + intros v c Hv. inversion Hv. reflexivity.
  - split; [ | split ].
    + intros c. split; constructor.
    + intros tag fields fields' Hfields. constructor. exact Hfields.
    + intros elements elements' Helements. constructor. exact Helements.
  - intros v v' Hv.
    split;
      [ | split; [ | split; [ | split; [ | split; [ | split; [ | split ] ] ] ] ]
      ].
    + intros tag fields. exact (block_of_lifted v v' tag fields Hv).
    + intros c. exact (boxed_number_of_lifted v v' c Hv).
    + intros bytes. exact (string_of_lifted v v' bytes Hv).
    + intros Hclosure.
      destruct (closure_of statics v) as [[called cslots] | ] eqn:Hc;
        [ | congruence ].
      destruct (closure_of_lifted v v' called cslots Hv Hc)
        as (cslots' & -> & _).
      discriminate.
    + pose proof (partial_application_lifted statics' v v' Hv) as Hpartial.
      destruct (partial_application_of v) as [[[closure applied] arity] | ];
        [ | congruence ].
      destruct Hpartial as (closure' & applied' & -> & _). discriminate.
    + lifted_cases Hv; simpl; try reflexivity.
      * destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
          [ | discriminate ].
        destruct (Hsymbols s allocated Hs) as (allocated' & -> & _).
        reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite Hlookup. reflexivity.
      * rewrite (Hbound fslot s (lookup_in _ _ _ Hsym)). reflexivity.
      * rewrite Hlookup. reflexivity.
    + intros [s0 c] Hs0.
      apply imported_of_symbol in Hs0 as [-> Hs0].
      inversion Hv; subst. simpl.
      destruct (Hsymbols s0 _ Hs0) as (allocated' & -> & Hallocated).
      inversion Hallocated. reflexivity.
    + intros elements. exact (array_of_lifted v v' elements Hv).
Qed.

Lemma eval_prim_lifted (p : prim) (vs vs' : list value) (v : value) :
  Forall2 (value_lifted statics') vs vs' ->
  eval_prim statics p vs = Some v ->
  ∃ v' : value, eval_prim statics' p vs' = Some v' ∧ value_lifted statics' v v'.
Proof.
  exact
    (eval_prim_related (value_lifted statics') statics statics' p vs vs' v
       value_lifted_respects).
Qed.

End Observations.

(* ======================================================================== *)
(* Evaluation in lifted environments                                        *)
(* ======================================================================== *)

(* Every variable of the first environment is bound in the second, to
   its value lifted. *)
Definition env_lifted (statics' : static_env) (vars vars' : env) : Prop :=
  ∀ (x : variable) (v : value),
    lookup x vars = Some v ->
    ∃ v' : value, lookup x vars' = Some v' ∧ value_lifted statics' v v'.

Lemma env_lifted_extend (statics' statics'' : static_env) (vars vars' : env) :
  statics_extend statics' statics'' ->
  env_lifted statics' vars vars' ->
  env_lifted statics'' vars vars'.
Proof.
  intros Hextend Hvars x v Hx. destruct (Hvars x v Hx) as (v' & Hx' & Hv).
  exists v'. split; [ exact Hx' | ].
  exact (value_lifted_extend _ _ Hextend _ _ Hv).
Qed.

Lemma env_lifted_cons (statics' : static_env) (x : variable) (v v' : value)
    (vars vars' : env) :
  value_lifted statics' v v' ->
  env_lifted statics' vars vars' ->
  env_lifted statics' ((x, v) :: vars) ((x, v') :: vars').
Proof.
  intros Hv Hvars y w. simpl.
  destruct (Pos.eqb y x); [ intros [= <-]; eauto | apply Hvars ].
Qed.

Lemma bind_params_lifted (statics' : static_env) (params : list variable) :
  ∀ (vs vs' : list value) (vars vars' bound : env),
    Forall2 (value_lifted statics') vs vs' ->
    env_lifted statics' vars vars' ->
    bind_params params vs vars = Some bound ->
    ∃ bound' : env,
      bind_params params vs' vars' = Some bound'
      ∧ env_lifted statics' bound bound'.
Proof.
  induction params as [ | p params IH]; intros vs vs' vars vars' bound Hvs;
    destruct Hvs as [ | v v' vs vs' Hv Hvs]; simpl; try discriminate.
  - intros Hvars [= <-]. eauto.
  - intros Hvars. apply (IH vs vs' _ _ bound Hvs).
    apply env_lifted_cons; assumption.
Qed.

Lemma eval_simple_lifted (statics' : static_env) (vars vars' : env)
    (s : simple) (v : value) :
  env_lifted statics' vars vars' ->
  eval_simple vars s = Some v ->
  ∃ v' : value, eval_simple vars' s = Some v' ∧ value_lifted statics' v v'.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - apply Hvars.
  - intros [= <-]. eexists. split; [ reflexivity | apply Lifted_symbol ].
  - intros [= <-]. eexists. split; [ reflexivity | apply Lifted_const ].
Qed.

Lemma eval_simples_lifted (statics' : static_env) (vars vars' : env)
    (args : list simple) :
  env_lifted statics' vars vars' ->
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' args = Some vs'
      ∧ Forall2 (value_lifted statics') vs vs'.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. exists []. auto.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    intros [= <-].
    destruct (eval_simple_lifted statics' vars vars' s v Hvars Hs)
      as (v' & -> & Hv).
    destruct (IH rest eq_refl) as (rest' & -> & Hrest).
    exists (v' :: rest'). auto.
Qed.

Lemma eval_value_slots_lifted (statics' : static_env) (vars vars' : env)
    (slots : list (value_slot * simple)) :
  env_lifted statics' vars vars' ->
  ∀ evaluated : list (value_slot * value),
    eval_value_slots vars slots = Some evaluated ->
    ∃ evaluated' : list (value_slot * value),
      eval_value_slots vars' slots = Some evaluated'
      ∧ Forall2 (slot_lifted (value_lifted statics')) evaluated evaluated'.
Proof.
  intros Hvars. induction slots as [ | [slot s] slots IH]; intros evaluated;
    simpl.
  - intros [= <-]. exists []. auto.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [rest | ]; [ | discriminate ].
    intros [= <-].
    destruct (eval_simple_lifted statics' vars vars' s v Hvars Hs)
      as (v' & -> & Hv).
    destruct (IH rest eq_refl) as (rest' & -> & Hrest).
    exists ((slot, v') :: rest'). split; [ reflexivity | ].
    constructor; [ split; [ reflexivity | exact Hv ] | exact Hrest ].
Qed.

(* A defining expression that has a value in the first run has one in
   the second, lifted. *)
Lemma eval_named_lifted (statics statics' : static_env) (vars vars' : env)
    (d : named) (v : value) :
  symbols_lifted statics statics' ->
  env_lifted statics' vars vars' ->
  eval_named statics vars d = Some v ->
  ∃ v' : value,
    eval_named statics' vars' d = Some v' ∧ value_lifted statics' v v'.
Proof.
  intros Hstatics Hvars.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl.
  - exact (eval_simple_lifted statics' vars vars' s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    destruct (eval_simples_lifted statics' vars vars' args Hvars vs Hargs)
      as (vs' & -> & Hvs).
    exact (eval_prim_lifted statics statics' Hstatics p vs vs' v Hvs).
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_lifted statics' vars vars' slots Hvars vs
                Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_closure, Hvs.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_lifted statics' vars vars' slots Hvars vs
                Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_closure_of_set, Hvs.
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct (eval_simple_lifted statics' vars vars' closure w Hvars Hclosure)
      as (w' & -> & Hw).
    simpl.
    destruct (closure_of statics w) as [[called cslots] | ] eqn:Hc;
      [ | discriminate ].
    destruct (closure_of_lifted statics statics' Hstatics w w' called cslots
                Hw Hc)
      as (cslots' & -> & Hcslots).
    simpl. exact (lookup_slots_lifted statics' slot cslots cslots' v Hcslots).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct (eval_simple_lifted statics' vars vars' closure w Hvars Hclosure)
      as (w' & -> & Hw).
    exact (project_value_slot_lifted statics statics' Hstatics pfrom slot w
             w' v Hw).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hclosure;
      [ | discriminate ].
    destruct (eval_simple_lifted statics' vars vars' closure w Hvars Hclosure)
      as (w' & -> & Hw).
    exact (project_function_slot_lifted statics statics' Hstatics move_from
             move_to w w' v Hw).
Qed.

(* What the machine runs a defining expression to, on its heap, it
   runs it to in the second run, on a heap alike: to a value lifted,
   and heaps alike after. A primitive does, by the law of the
   primitives on heaps alike (FlambdaHeapSpec.exec_prim_related_statement);
   nothing else looks at the heap. The records of the two runs are not
   compared. *)
Lemma exec_named_lifted (statics statics' : static_env) (hp hp' hp1 : heap)
    (rd rd1 rd' : read_record) (vars vars' : env) (d : named) (v : value) :
  symbols_lifted statics statics' ->
  env_lifted statics' vars vars' ->
  heap_alike (value_lifted statics') hp hp' ->
  exec_named statics hp rd vars d = Some (v, hp1, rd1) ->
  ∃ (v' : value) (hp1' : heap),
    exec_named statics' hp' rd' vars' d = Some (v', hp1', rd')
    ∧ value_lifted statics' v v'
    ∧ heap_alike (value_lifted statics') hp1 hp1'.
Proof.
  intros Hstatics Hvars Hheap Hexec.
  assert (Hcases :
            (∃ (p : prim) (args : list simple), d = Prim p args)
            ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d; try (right; intros ? ?; discriminate).
    left. eexists. eexists. reflexivity. }
  destruct Hcases as [(p & args & ->) | Hnot].
  - apply exec_named_prim_some in Hexec as (vs & Hargs & Hprim & _).
    destruct (eval_simples_lifted statics' vars vars' args Hvars vs Hargs)
      as (vs' & Hargs' & Hvs).
    destruct
      (exec_prim_related (value_lifted statics') statics statics' hp hp' hp1
         p vs vs' v (value_lifted_respects statics statics' Hstatics)
         (value_lifted_locations statics') Hheap Hvs Hprim)
      as (v' & hp1' & Hprim' & Hv & Hheap1).
    exists v', hp1'. split; [ | split; assumption ].
    apply exec_named_prim_some. exists vs'. auto.
  - rewrite (exec_named_not_prim _ _ _ _ _ Hnot) in Hexec.
    destruct (eval_named statics vars d) as [w | ] eqn:Hw; [ | discriminate ].
    injection Hexec as <- <- <-.
    destruct (eval_named_lifted statics statics' vars vars' d w Hstatics Hvars
                Hw)
      as (v' & Hd' & Hv).
    exists v', hp'. split; [ | split; assumption ].
    rewrite (exec_named_not_prim _ _ _ _ _ Hnot), Hd'. reflexivity.
Qed.

Lemma eval_static_const_lifted (statics' : static_env) (vars vars' : env)
    (const : static_const) (allocated : statically_allocated_value) :
  env_lifted statics' vars vars' ->
  eval_static_const vars const = Some allocated ->
  ∃ allocated' : statically_allocated_value,
    eval_static_const vars' const = Some allocated'
    ∧ allocated_lifted statics' allocated allocated'.
Proof.
  intros Hvars.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_lifted statics' vars vars' slots Hvars vs
                Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_static_closure, Hvs.
  - destruct (lookup fslot decls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    destruct (eval_value_slots_lifted statics' vars vars' slots Hvars vs
                Hslots)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_static_closure_of_set, Hvs.
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    destruct (eval_simples_lifted statics' vars vars' fields Hvars vs Hfields)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_static_block, Hvs.
  - destruct (eval_simple vars contents) as [w | ] eqn:Hcontents;
      [ | discriminate ].
    destruct (eval_simple_lifted statics' vars vars' contents w Hvars
                Hcontents)
      as (w' & -> & Hw).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_static_boxed_number, Hw.
  - intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_static_string.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    destruct
      (eval_simples_lifted statics' vars vars' elements Hvars vs Helements)
      as (vs' & -> & Hvs).
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Lifted_static_immutable_array, Hvs.
Qed.

(* ======================================================================== *)
(* Static bindings                                                          *)
(* ======================================================================== *)

(* A group that the machine does not refuse for binding a symbol twice:
   its symbols are distinct, and none is bound. *)
Lemma no_rebinding (statics' : static_env) (bindings : list static_binding) :
  rebinds symbols_of_binding (symbol_bound statics') bindings = false ->
  distinct (flat_map symbols_of_binding bindings) = true
  ∧ ∀ s : symbol,
      s ∈ flat_map symbols_of_binding bindings ->
      lookup s (static_values statics') = None.
Proof.
  unfold rebinds. intros Hrebinds.
  apply orb_false_iff in Hrebinds as [Hdistinct Hbound].
  apply negb_false_iff in Hdistinct. split; [ exact Hdistinct | ].
  intros s Hin.
  destruct (lookup s (static_values statics')) eqn:Hs; [ | reflexivity ].
  exfalso.
  assert (Hexists :
    existsb (symbol_bound statics') (flat_map symbols_of_binding bindings)
    = true).
  { apply existsb_exists. exists s. split; [ exact Hin | ].
    unfold symbol_bound. rewrite Hs. reflexivity. }
  exact (eq_true_false_abs _ Hexists Hbound).
Qed.

(* A symbol that is not bound may be bound: what was bound stays so. *)
Lemma statics_extend_symbol (statics' : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  lookup s (static_values statics') = None ->
  statics_extend statics'
    (Static_env (static_code statics')
       ((s, allocated) :: static_values statics')).
Proof.
  intros Hfresh. split; simpl; [ auto | ].
  intros s0 a Hs0. exact (lookup_fresh s0 s a allocated _ Hfresh Hs0).
Qed.

(* A group of bindings that the first run evaluates, the second does,
   given that it binds none of the symbols already: the same group, in
   lifted environments, gives lifted static environments. *)
Lemma eval_static_bindings_lifted (vars vars' : env) :
  ∀ (bindings : list static_binding) (statics statics' extended : static_env),
    env_lifted statics' vars vars' ->
    statics_lifted statics statics' ->
    distinct (flat_map symbols_of_binding bindings) = true ->
    (∀ s : symbol,
      s ∈ flat_map symbols_of_binding bindings ->
      lookup s (static_values statics') = None) ->
    eval_static_bindings vars bindings statics = Some extended ->
    ∃ extended' : static_env,
      eval_static_bindings vars'
        (map (bind_my_closures_in_binding known) bindings) statics'
      = Some extended'
      ∧ statics_lifted extended extended'.
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics statics' extended Hvars Hstatics Hdistinct Hfresh; simpl.
  - intros [= <-]. eauto.
  - destruct binding as [id c | id | sym const];
      simpl in Hdistinct, Hfresh |- *.
    + (* Code: bound alike in both. *)
      destruct (lookup id (static_code statics)) eqn:Hid; [ discriminate | ].
      assert (Hid' : lookup id (static_code statics') = None)
        by (rewrite (proj1 Hstatics id), Hid; reflexivity).
      rewrite Hid'.
      assert (Hextend :
        statics_extend statics'
          (Static_env
             ((id, bind_my_closure known id c) :: static_code statics')
             (static_values statics'))).
      { split; simpl; [ | auto ]. intros id0 c0 Hid0.
        exact (lookup_fresh id0 id c0 _ _ Hid' Hid0). }
      apply IH;
        [ exact (env_lifted_extend _ _ _ _ Hextend Hvars) |
        | exact Hdistinct | exact Hfresh ].
      split; simpl.
      * intros id0. destruct (Pos.eqb_spec id0 id) as [-> | _];
          [ reflexivity | apply (proj1 Hstatics) ].
      * intros s a Hs. destruct (proj2 Hstatics s a Hs) as (a' & Hs' & Ha).
        exists a'. split; [ exact Hs' | ].
        exact (allocated_lifted_extend _ _ _ _ Hextend Ha).
    + apply IH; assumption.
    + (* A symbol: not bound in the second run either. *)
      destruct (lookup sym (static_values statics)) eqn:Hsym;
        [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
        [ | discriminate ].
      simpl. rewrite (Hfresh sym (or_introl eq_refl)).
      destruct (eval_static_const_lifted statics' vars vars' const allocated
                  Hvars Hconst)
        as (allocated' & -> & Hallocated).
      simpl. apply andb_true_iff in Hdistinct as [Hnotin Hdistinct].
      pose proof (statics_extend_symbol statics' sym allocated'
        (Hfresh sym (or_introl eq_refl))) as Hextend.
      apply IH;
        [ exact (env_lifted_extend _ _ _ _ Hextend Hvars) |
        | exact Hdistinct | ].
      * split; [ exact (proj1 Hstatics) | ].
        intros s a. cbn [static_values]. rewrite lookup_cons.
        destruct (Pos.eqb_spec s sym) as [-> | Hother].
        -- intros [= <-]. exists allocated'.
           split; [ apply lookup_head | ].
           exact (allocated_lifted_extend _ _ _ _ Hextend Hallocated).
        -- intros Hs. destruct (proj2 Hstatics s a Hs) as (a' & Hs' & Ha).
           exists a'. split; [ rewrite lookup_other; assumption | ].
           exact (allocated_lifted_extend _ _ _ _ Hextend Ha).
      * intros s Hin. cbn [static_values].
        rewrite lookup_other; [ apply Hfresh; right; exact Hin | ].
        intros ->. apply negb_true_iff in Hnotin.
        apply (proj2 (among_In _ _)) in Hin.
        exact (eq_true_false_abs _ Hin Hnotin).
Qed.

(* The bindings of a static set of closures bind its symbols. *)
Lemma bind_closures_symbols (syms : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * simple)) :
  flat_map symbols_of_binding (bind_closures syms decls slots) = map snd syms.
Proof.
  unfold bind_closures. generalize syms at 1 as all.
  intros all. induction syms as [ | [slot s] syms IH]; simpl;
    [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

(* They are evaluated, where the value slots have values, every function
   slot that has a symbol is one of the set, and the symbols are distinct
   and not bound. Code is not touched. [all] is the whole set, kept apart
   from the symbols still to be bound. *)
Lemma eval_closure_bindings (vars' : env)
    (all : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) :
  eval_value_slots vars' slots = Some slot_values ->
  ∀ (rest : list (function_slot * symbol)) (statics' : static_env),
    (∀ (slot : function_slot) (s : symbol),
      (slot, s) ∈ rest -> lookup slot decls <> None) ->
    distinct (map snd rest) = true ->
    (∀ s : symbol,
      s ∈ map snd rest -> lookup s (static_values statics') = None) ->
    ∃ extended' : static_env,
      eval_static_bindings vars'
        (map
           (fun '(slot, s) =>
              Bind_symbol s
                (Static_const_closure_of_set slot all decls slots))
           rest)
        statics'
      = Some extended'
      ∧ static_code extended' = static_code statics'.
Proof.
  intros Hslots.
  induction rest as [ | [slot s] rest IH]; intros statics' Hdecls Hdistinct
    Hfresh; simpl.
  - eauto.
  - simpl in Hdistinct, Hfresh.
    rewrite (Hfresh s (or_introl eq_refl)).
    destruct (lookup slot decls) as [code | ] eqn:Hcode;
      [ | destruct (Hdecls slot s (or_introl eq_refl) Hcode) ].
    simpl. rewrite Hslots. simpl.
    apply andb_true_iff in Hdistinct as [Hnotin Hdistinct].
    destruct
      (IH
         (Static_env (static_code statics')
            ((s, Static_closure_of_set slot all decls slot_values)
             :: static_values statics')))
      as (extended' & Heval & Hcode');
      [ intros slot' s' Hin; apply (Hdecls slot' s'); right; exact Hin
      | exact Hdistinct | | eauto ].
    intros s' Hin. cbn [static_values].
    rewrite lookup_other; [ apply Hfresh; right; exact Hin | ].
    intros ->. apply negb_true_iff in Hnotin.
    apply (proj2 (among_In _ _)) in Hin.
    exact (eq_true_false_abs _ Hin Hnotin).
Qed.

(* The three [Let]s that are lifted. The second run binds the symbols,
   which it can as none is bound, and the value the first run made is
   the symbol's, lifted. *)
Lemma lift_named_lifted (target : lift_to) (d : named)
    (bindings : list static_binding) (s : symbol)
    (statics statics' : static_env) (vars vars' : env) (v : value) :
  lift_named target d = Some (bindings, s) ->
  env_lifted statics' vars vars' ->
  eval_named statics vars d = Some v ->
  distinct (flat_map symbols_of_binding bindings) = true ->
  (∀ s0 : symbol,
    s0 ∈ flat_map symbols_of_binding bindings ->
    lookup s0 (static_values statics') = None) ->
  ∃ extended' : static_env,
    eval_static_bindings vars' bindings statics' = Some extended'
    ∧ static_code extended' = static_code statics'
    ∧ value_lifted extended' v (Value_symbol s).
Proof.
  intros Hlift Hvars Heval Hdistinct Hfresh.
  destruct target as [sym | syms].
  - destruct d as [ | p args | code slots | | | | ]; simpl in Hlift;
      try discriminate.
    2: { (* A closure of a set of one function. *)
      injection Hlift as <- <-. simpl in Heval.
      destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
        [ | discriminate ].
      injection Heval as <-.
      destruct (eval_value_slots_lifted statics' vars vars' slots Hvars vs
                  Hslots)
        as (vs' & Hslots' & Hvs).
      simpl. rewrite (Hfresh sym (or_introl eq_refl)), Hslots'. simpl.
      eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
      apply (Lifted_to_unnamed_closure _ code vs vs' sym);
        [ simpl; apply lookup_head | ].
      refine (slots_lifted_extend _ _ _ _ _ Hvs).
      exact (statics_extend_symbol statics' sym _
               (Hfresh sym (or_introl eq_refl))). }
    destruct p as [p | p | p | p | rp]; simpl in Hlift; try discriminate.
    + (* A boxed number. *)
      destruct p; simpl in Hlift; try discriminate.
      destruct args as [ | contents [ | arg args]]; simpl in Hlift;
        try discriminate.
      injection Hlift as <- <-. simpl in Heval.
      destruct (eval_simple vars contents) as [w | ] eqn:Hcontents;
        [ | discriminate ].
      rewrite eval_prim_unary in Heval. simpl in Heval.
      destruct w as [number | | | | | | | | ]; try discriminate.
      destruct (is_number_of_kind kind number); [ | discriminate ].
      injection Heval as <-.
      destruct (eval_simple_lifted statics' vars vars' contents _ Hvars
                  Hcontents)
        as (w' & Hcontents' & Hw).
      inversion Hw; subst w'.
      simpl. rewrite (Hfresh sym (or_introl eq_refl)), Hcontents'. simpl.
      eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
      apply Lifted_to_boxed_number. simpl. apply lookup_head.
    + (* A block, or an immutable array. *)
      destruct p as [[tag] | kind]; simpl in Hlift.
      * injection Hlift as <- <-. simpl in Heval.
        destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
          [ | discriminate ].
        rewrite eval_prim_variadic in Heval. simpl in Heval.
        destruct
          (forallb (fun v => kind_eqb (kind_of_value v) Value) vs
           && forallb (bound_if_symbol statics) vs);
          [ | discriminate ].
        injection Heval as <-.
        destruct (eval_simples_lifted statics' vars vars' args Hvars vs Hargs)
          as (vs' & Hargs' & Hvs).
        simpl. rewrite (Hfresh sym (or_introl eq_refl)), Hargs'. simpl.
        eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
        apply (Lifted_to_block _ tag vs vs' sym);
          [ simpl; apply lookup_head | ].
        refine (values_lifted_extend _ _ _ _ _ Hvs).
        exact (statics_extend_symbol statics' sym _
                 (Hfresh sym (or_introl eq_refl))).
      * (* The array that the [Let] made has the arguments for its
           elements, and the symbol is bound to an array of what the
           arguments are in the second run. *)
        injection Hlift as <- <-. simpl in Heval.
        destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
          [ | discriminate ].
        rewrite eval_prim_variadic in Heval. simpl in Heval.
        unfold array_made in Heval.
        destruct (forallb (array_element_ok kind) vs); [ | discriminate ].
        simpl in Heval.
        destruct (forallb (bound_if_symbol statics) vs); [ | discriminate ].
        injection Heval as <-.
        destruct (eval_simples_lifted statics' vars vars' args Hvars vs Hargs)
          as (vs' & Hargs' & Hvs).
        simpl. rewrite (Hfresh sym (or_introl eq_refl)), Hargs'. simpl.
        eexists. split; [ reflexivity | ]. split; [ reflexivity | ].
        apply (Lifted_to_array _ vs vs' sym); [ simpl; apply lookup_head | ].
        refine (values_lifted_extend _ _ _ _ _ Hvs).
        exact (statics_extend_symbol statics' sym _
                 (Hfresh sym (or_introl eq_refl))).
  - (* A closure of a set. *)
    destruct d as [ | | | fslot decls slots | | | ]; simpl in Hlift;
      try discriminate.
    destruct (lookup fslot syms) as [s1 | ] eqn:Hsym; [ | discriminate ].
    destruct (list_eq_dec Pos.eq_dec (map fst syms) (map fst decls))
      as [Hsyms | ];
      [ | discriminate ].
    injection Hlift as <- <-. simpl in Heval.
    destruct (lookup fslot decls) as [code | ]; [ | discriminate ].
    simpl in Heval.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    injection Heval as <-.
    destruct (eval_value_slots_lifted statics' vars vars' slots Hvars vs
                Hslots)
      as (vs' & Hslots' & Hvs).
    rewrite bind_closures_symbols in Hdistinct, Hfresh.
    destruct (eval_closure_bindings vars' syms decls slots vs' Hslots' syms
                statics')
      as (extended' & Hextended & Hcode);
      [ | exact Hdistinct | exact Hfresh | ].
    { intros slot s0 Hin Hnone.
      apply (lookup_none_not_in slot decls Hnone).
      pose proof (in_map fst syms (slot, s0) Hin) as Hslot.
      rewrite Hsyms in Hslot. exact Hslot. }
    exists extended'. split; [ exact Hextended | ]. split; [ exact Hcode | ].
    apply (Lifted_to_closure extended' fslot decls vs vs' s1 syms Hsym Hsyms).
    + exact (bind_closures_binds vars' syms decls slots vs' statics' extended'
               Hslots' Hextended).
    + refine (slots_lifted_extend _ _ _ _ _ Hvs).
      exact (eval_static_bindings_extend vars' _ statics' extended' Hextended).
Qed.

(* A [Let] that is lifted reads no cell: the machine runs it as
   [eval_named] says and leaves the heap. *)
Lemma lift_named_reads_no_heap (target : lift_to) (d : named)
    (bindings : list static_binding) (s : symbol) :
  lift_named target d = Some (bindings, s) -> named_reads_heap d = false.
Proof.
  intros Hlift. destruct target as [sym | syms].
  - destruct d as [ | p args | code slots | | | | ]; simpl in Hlift;
      try discriminate.
    2: reflexivity.
    destruct p as [p | p | p | p | rp]; simpl in Hlift; try discriminate.
    + destruct p; simpl in Hlift; try discriminate. reflexivity.
    + reflexivity.
  - destruct d as [ | | | fslot decls slots | | | ]; simpl in Hlift;
      try discriminate.
    reflexivity.
Qed.

(* The bindings that stand for a lifted [Let] bind symbols only, so
   binding [my_closure] in code changes nothing in them. *)
Lemma lift_named_no_code (target : lift_to) (d : named)
    (bindings : list static_binding) (s : symbol) :
  lift_named target d = Some (bindings, s) ->
  map (bind_my_closures_in_binding known) bindings = bindings.
Proof.
  intros Hlift. destruct target as [sym | syms].
  - destruct d as [ | p args | code slots | | | | ]; simpl in Hlift;
      try discriminate.
    2: { injection Hlift as <- <-. reflexivity. }
    destruct p as [p | p | p | p | rp]; simpl in Hlift; try discriminate.
    + destruct p; simpl in Hlift; try discriminate.
      destruct args as [ | contents [ | arg args]]; simpl in Hlift;
        try discriminate.
      injection Hlift as <- <-. reflexivity.
    + destruct p as [[tag] | kind]; simpl in Hlift;
        injection Hlift as <- <-; reflexivity.
  - destruct d as [ | | | fslot decls slots | | | ]; simpl in Hlift;
      try discriminate.
    destruct (lookup fslot syms) as [s1 | ]; [ | discriminate ].
    destruct (list_eq_dec Pos.eq_dec (map fst syms) (map fst decls));
      [ | discriminate ].
    injection Hlift as <- <-. unfold bind_closures. rewrite map_map.
    apply map_ext. intros [slot' s']. reflexivity.
Qed.

(* ======================================================================== *)
(* Continuations and states                                                 *)
(* ======================================================================== *)

(* With no [Let] to lift, the term is left as it is. So code and the
   handlers of recursive continuations, which [lift_expr] does not go
   into, are themselves lifted. *)
Fixpoint lift_expr_nil (at_path : path) (e : expr) {struct e} :
  lift_expr [] at_path e = e
with lift_handler_nil (at_path : path) (h : continuation_handler)
    {struct h} :
  lift_handler [] at_path h = h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite lift_expr_nil. reflexivity.
    + rewrite lift_expr_nil. reflexivity.
    + rewrite lift_handler_nil, lift_expr_nil. reflexivity.
    + rewrite lift_expr_nil. reflexivity.
  - destruct h as [params metadata handler]. simpl. rewrite lift_expr_nil.
    reflexivity.
Qed.

Definition bind_my_closures_in_handlers
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k, h) => (k, bind_my_closures_in_handler known h)) handlers.

(* A term with no static binding in it binds no code, so [my_closure]
   is bound nowhere in it: it is left as it is. *)
Fixpoint bind_my_closures_no_static (e : expr) {struct e} :
  binds_no_static e = true -> bind_my_closures known e = e.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; reflexivity).
  - intros H. rewrite (bind_my_closures_no_static body H). reflexivity.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros H.
    apply andb_true_iff in H as [Hh Hb].
    rewrite (bind_my_closures_no_static handler Hh),
      (bind_my_closures_no_static body Hb).
    reflexivity.
  - intros H. apply andb_true_iff in H as [Hhs Hb].
    rewrite (bind_my_closures_no_static body Hb). f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhs. apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (bind_my_closures_no_static handler Hh), (IH Hrest).
    reflexivity.
Qed.

(* A continuation of the first run and one of the second: the same way
   out, or closures of a handler and of that handler rewritten, with
   lifted environments, or what is left of the same over-application,
   with lifted arguments and continuations. The handlers of a recursive
   group are not gone into for [Let]s to lift. *)
#[local] Set Warnings "-register-all".
Inductive cont_lifted (statics' : static_env)
    : cont_value -> cont_value -> Prop :=
  | Outer_lifted (k : continuation) : cont_lifted statics' (Outer k) (Outer k)
  | Closure_lifted (lets : list (path * lift_to)) (at_path : path)
      (h : continuation_handler) (vars vars' : env) (conts conts' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    env_lifted statics' vars vars' ->
    Forall2
      (fun kc kc' =>
         fst kc = fst kc' ∧ cont_lifted statics' (snd kc) (snd kc'))
      conts conts' ->
    cont_lifted statics' (Closure h vars conts recursive)
      (Closure
         (bind_my_closures_in_handler known (lift_handler lets at_path h))
         vars' conts' (bind_my_closures_in_handlers recursive))
  | Extra_args_lifted (c c' : cont_value) (extra extra' : list value) :
    cont_lifted statics' c c' ->
    Forall2 (value_lifted statics') extra extra' ->
    cont_lifted statics' (Extra_args c extra) (Extra_args c' extra')
  | Over_application_lifted (remaining remaining' : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    Forall2 (value_lifted statics') remaining remaining' ->
    cont_lifted statics' return_to return_to' ->
    cont_lifted statics' exn_to exn_to' ->
    cont_lifted statics'
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining' remaining_arity return_arity return_to'
         exn_to').

Definition conts_lifted (statics' : static_env) (conts conts' : cont_env)
    : Prop :=
  Forall2
    (fun kc kc' => fst kc = fst kc' ∧ cont_lifted statics' (snd kc) (snd kc'))
    conts conts'.

(* What may follow a unit is the same in the two runs: it has no [Let]
   that is lifted and no code. *)
Lemma observer_lifted (statics' : static_env) (slots : list value_slot)
    (c : cont_value) :
  observer slots c -> cont_lifted statics' c c.
Proof.
  destruct c
    as [ [params metadata body] vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction.
  - destruct vars; [ | contradiction ].
    destruct conts; [ | contradiction ].
    destruct recursive; [ | contradiction ].
    intros (Hno & _ & _).
    assert (H :
      cont_lifted statics' (Closure (Handler params metadata body) [] [] [])
        (Closure
           (bind_my_closures_in_handler known
              (lift_handler [] root (Handler params metadata body)))
           [] [] (bind_my_closures_in_handlers []))).
    { apply Closure_lifted; [ intros x v [=] | constructor ]. }
    rewrite lift_handler_nil in H. simpl in H.
    rewrite (bind_my_closures_no_static body Hno) in H. exact H.
  - intros _. apply Outer_lifted.
Qed.

Lemma observers_lifted (statics' : static_env) (slots : list value_slot)
    (conts : cont_env) :
  observers slots conts -> conts_lifted statics' conts conts.
Proof.
  unfold observers, conts_lifted.
  induction conts as [ | [k c] conts IH]; intros H; [ constructor | ].
  inversion H as [ | kc rest Hc Hrest]; subst.
  constructor; [ | exact (IH Hrest) ].
  split; [ reflexivity | exact (observer_lifted statics' slots c Hc) ].
Qed.

Fixpoint cont_lifted_extend (statics' statics'' : static_env)
    (Hextend : statics_extend statics' statics'') (c c' : cont_value)
    (H : cont_lifted statics' c c') {struct H} :
  cont_lifted statics'' c c'.
Proof.
  destruct H
    as [ k | lets at_path h vars vars' conts conts' recursive Hvars Hconts
       | c0 c0' extra extra' Hc0 Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn Hexn ].
  - apply Outer_lifted.
  - apply Closure_lifted;
      [ exact (env_lifted_extend _ _ _ _ Hextend Hvars) | ].
    clear Hvars. revert conts conts' Hconts. fix IH 3.
    intros conts conts' [ | kc kc' rest rest' [Hkey Hhead] Htail];
      constructor;
      [ split;
        [ exact Hkey | exact (cont_lifted_extend _ _ Hextend _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply Extra_args_lifted;
      [ exact (cont_lifted_extend _ _ Hextend _ _ Hc0)
      | exact (values_lifted_extend _ _ _ _ Hextend Hextra) ].
  - apply Over_application_lifted;
      [ exact (values_lifted_extend _ _ _ _ Hextend Hremaining)
      | exact (cont_lifted_extend _ _ Hextend _ _ Hreturn)
      | exact (cont_lifted_extend _ _ Hextend _ _ Hexn) ].
Qed.

Lemma with_extra_args_lifted (statics' : static_env) (c c' : cont_value)
    (extra extra' : list value) :
  cont_lifted statics' c c' ->
  Forall2 (value_lifted statics') extra extra' ->
  cont_lifted statics' (with_extra_args c extra) (with_extra_args c' extra').
Proof.
  intros Hc Hextra.
  destruct Hextra as [ | v v' extra extra' Hv Hextra]; [ exact Hc | ].
  apply Extra_args_lifted; [ exact Hc | constructor; assumption ].
Qed.

Lemma conts_lifted_extend (statics' statics'' : static_env)
    (conts conts' : cont_env) :
  statics_extend statics' statics'' ->
  conts_lifted statics' conts conts' ->
  conts_lifted statics'' conts conts'.
Proof.
  intros Hextend.
  induction 1 as [ | kc kc' conts conts' [Hkey Hc] _ IH]; constructor;
    [ split; [ exact Hkey | exact (cont_lifted_extend _ _ Hextend _ _ Hc) ]
    | exact IH ].
Qed.

Lemma resolve_lifted (statics' : static_env) (k : continuation)
    (conts conts' : cont_env) :
  conts_lifted statics' conts conts' ->
  cont_lifted statics' (resolve k conts) (resolve k conts').
Proof.
  unfold resolve.
  induction 1 as [ | [k1 c] [k1' c'] conts conts' [Hk Hc] _ IH]; simpl;
    [ apply Outer_lifted | ].
  simpl in Hk. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

(* [all] is the whole group, kept apart from the handlers still to be gone
   through. *)
Lemma rec_closures_lifted_aux (statics' : static_env) (vars vars' : env)
    (conts conts' : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  env_lifted statics' vars vars' ->
  conts_lifted statics' conts conts' ->
  conts_lifted statics'
    (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
    (map
       (fun '(k, h) =>
          (k, Closure h vars' conts' (bind_my_closures_in_handlers all)))
       (bind_my_closures_in_handlers handlers)).
Proof.
  intros Hvars Hconts.
  induction handlers as [ | [k h] handlers IH]; simpl; constructor;
    [ | exact IH ].
  split; [ reflexivity | ]. simpl.
  pose proof (Closure_lifted statics' [] root h vars vars' conts conts' all
    Hvars Hconts) as Hclosure.
  rewrite lift_handler_nil in Hclosure. exact Hclosure.
Qed.

Lemma rec_closures_lifted (statics' : static_env) (vars vars' : env)
    (conts conts' : cont_env)
    (handlers : list (continuation * continuation_handler)) :
  env_lifted statics' vars vars' ->
  conts_lifted statics' conts conts' ->
  conts_lifted statics' (rec_closures handlers vars conts ++ conts)
    (rec_closures (bind_my_closures_in_handlers handlers) vars' conts'
     ++ conts').
Proof.
  intros Hvars Hconts. apply Forall2_app; [ | exact Hconts ].
  apply rec_closures_lifted_aux; assumption.
Qed.

(* The first state and the second: the second expression is the first
   with [Let]s lifted and [my_closure] bound in code, the environments
   are lifted, and the heaps are alike up to [value_lifted]: a cell of
   the second holds a symbol where that of the first holds what a
   [Let] that is lifted made. The read records are not compared. *)
Inductive state_lifted : state -> state -> Prop :=
  | State_lifted (lets : list (path * lift_to)) (at_path : path) (e : expr)
      (vars vars' : env) (conts conts' : cont_env)
      (statics statics' : static_env) (hp hp' : heap)
      (rd rd' : read_record) :
    env_lifted statics' vars vars' ->
    conts_lifted statics' conts conts' ->
    statics_lifted statics statics' ->
    heap_alike (value_lifted statics') hp hp' ->
    state_lifted (State e vars conts statics hp rd)
      (State (bind_my_closures known (lift_expr lets at_path e)) vars' conts'
         statics' hp' rd').

(* The values at an exit are lifted, and the heaps are alike, under the
   static environment that the second run has reached. *)
Definition exit_lifted (last : state) (vs : list value) (hp : heap)
    (vs' : list value) (hp' : heap) : Prop :=
  Forall2 (value_lifted (state_statics last)) vs vs'
  ∧ heap_alike (value_lifted (state_statics last)) hp hp'.

(* What the second run answers to a result of the first: nothing is
   asked if the first is undefined; an exit is answered by an exit
   through the same continuation, with values lifted and a heap alike;
   a step, by a step and then none or more, to a state lifted. *)
Definition result_lifted (statics' : static_env)
    (result result' : step_result) : Prop :=
  match result with
  | Undefined => True
  | Exit k exit_vs hp _ =>
    ∃ (exit_vs' : list value) (hp' : heap) (rd' : read_record),
      result' = Exit k exit_vs' hp' rd'
      ∧ Forall2 (value_lifted statics') exit_vs exit_vs'
      ∧ heap_alike (value_lifted statics') hp hp'
  | Step t =>
    ∃ (n : nat) (u' t' : state),
      result' = Step u' ∧ run n u' = Step t' ∧ state_lifted t t'
  end.

(* The second run makes no closure of the code that [known] names but
   the symbols it gives: the invariant of FlambdaMyClosureProofs.v, for
   each pair of [known], of a value, of a static environment and of a
   continuation. A call may call a value that another call returned, or
   the closure that a partial application keeps, so the simulation has
   the invariant of every value of the second run, not of the callee of
   an [Apply] only. *)
Definition known_free (v' : value) : Prop :=
  ∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> value_free id v' = true.

Definition known_statics_ok (statics' : static_env) : Prop :=
  ∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> statics_ok id sym statics'.

Definition known_cont_ok (c' : cont_value) : Prop :=
  ∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> cont_ok id sym c'.

Definition known_state_ok (s' : state) : Prop :=
  ∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> state_ok id sym s'.

Lemma known_free_all (vs' : list value) :
  (∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> forallb (value_free id) vs' = true) ->
  Forall known_free vs'.
Proof.
  intros H. apply Forall_forall. intros v Hin id sym Hsym.
  specialize (H id sym Hsym). rewrite forallb_forall in H. exact (H v Hin).
Qed.

Lemma known_free_partial (callee : value) (args : list value)
    (args_arity : arity) :
  known_free (Value_partial_application callee args args_arity)
  <-> known_free callee ∧ Forall known_free args.
Proof.
  split.
  - intros H. split.
    + intros id sym Hsym.
      exact (proj1
               (proj1 (value_free_partial id callee args args_arity)
                  (H id sym Hsym))).
    + apply Forall_forall. intros v Hin id sym Hsym.
      pose proof
        (proj2
           (proj1 (value_free_partial id callee args args_arity)
              (H id sym Hsym)))
        as Hargs.
      rewrite Forall_forall in Hargs. exact (Hargs v Hin).
  - intros [Hcallee Hargs] id sym Hsym. apply value_free_partial.
    split; [ exact (Hcallee id sym Hsym) | ].
    apply Forall_forall. intros v Hin. rewrite Forall_forall in Hargs.
    exact (Hargs v Hin id sym Hsym).
Qed.

Lemma known_cont_ok_extra_args (c' : cont_value) (extra' : list value) :
  known_cont_ok (Extra_args c' extra') ->
  known_cont_ok c' ∧ Forall known_free extra'.
Proof.
  intros H. split.
  - intros id sym Hsym. pose proof (H id sym Hsym) as Hok.
    inversion Hok. assumption.
  - apply known_free_all. intros id sym Hsym.
    pose proof (H id sym Hsym) as Hok. inversion Hok. assumption.
Qed.

Lemma known_cont_ok_over_application (remaining' : list value)
    (remaining_arity return_arity : arity)
    (return_to' exn_to' : cont_value) :
  known_cont_ok
    (Over_application remaining' remaining_arity return_arity return_to'
       exn_to') ->
  Forall known_free remaining' ∧ known_cont_ok return_to'.
Proof.
  intros H. split.
  - apply known_free_all. intros id sym Hsym.
    pose proof (H id sym Hsym) as Hok. inversion Hok. assumption.
  - intros id sym Hsym. pose proof (H id sym Hsym) as Hok.
    inversion Hok. assumption.
Qed.

(* The closure that the second run calls, lifted from the one that the
   first calls, is the symbol that [known] gives its code, if it gives
   one. *)
Definition called_lifted (statics' : static_env) (closure closure' : value)
    (called : code_id) : Prop :=
  value_lifted statics' closure closure'
  ∧ ∀ sym : symbol,
      lookup called known = Some sym -> closure' = Value_symbol sym.

(* Two calls of one shape: the closures called, the arguments and those
   that remain are lifted, and the code of the second is that of the
   first with [my_closure] bound. *)
Inductive shape_lifted (statics' : static_env)
    : call_shape -> call_shape -> Prop :=
  | Full_call_lifted (closure closure' : value) (called : code_id) (c : code)
      (vs vs' : list value) (args_arity : arity) :
    called_lifted statics' closure closure' called ->
    Forall2 (value_lifted statics') vs vs' ->
    shape_lifted statics' (Full_call closure c vs args_arity)
      (Full_call closure' (bind_my_closure known called c) vs' args_arity)
  | Over_call_lifted (closure closure' : value) (called : code_id) (c : code)
      (vs vs' remaining remaining' : list value)
      (args_arity remaining_arity : arity) :
    called_lifted statics' closure closure' called ->
    Forall2 (value_lifted statics') vs vs' ->
    Forall2 (value_lifted statics') remaining remaining' ->
    shape_lifted statics'
      (Over_call closure c vs remaining args_arity remaining_arity)
      (Over_call closure' (bind_my_closure known called c) vs' remaining'
         args_arity remaining_arity)
  | Partial_call_lifted (partial partial' : value) :
    value_lifted statics' partial partial' ->
    shape_lifted statics' (Partial_call partial) (Partial_call partial').

(* Binding [my_closure] leaves the arities of code as they are. *)
Lemma bind_my_closure_params_arity (id : code_id) (c : code) :
  code_params_arity (bind_my_closure known id c) = code_params_arity c.
Proof.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  destruct (lookup id known); [ destruct (among my_closure params) | ];
    reflexivity.
Qed.

Lemma bind_my_closure_result_arity (id : code_id) (c : code) :
  code_result_arity (bind_my_closure known id c) = code_result_arity c.
Proof.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  destruct (lookup id known); [ destruct (among my_closure params) | ];
    reflexivity.
Qed.

(* What the call of a closure is in the first run, the call of the
   closure lifted is in the second. *)
Lemma shape_of_closure_call_lifted (statics statics' : static_env)
    (closure closure' : value) (kind : call_kind) (vs vs' : list value)
    (args_arity return_arity : arity) (shape : call_shape) :
  statics_lifted statics statics' ->
  value_lifted statics' closure closure' ->
  Forall2 (value_lifted statics') vs vs' ->
  known_statics_ok statics' ->
  known_free closure' ->
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some shape ->
  ∃ shape' : call_shape,
    shape_of_closure_call statics' closure' kind vs' args_arity return_arity
    = Some shape'
    ∧ shape_lifted statics' shape shape'.
Proof.
  intros Hstatics Hclosure Hvs Hok Hfree. unfold shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slots] | ] eqn:Hc;
    [ | discriminate ].
  destruct (closure_of_lifted statics statics' (proj2 Hstatics) closure closure'
              called slots Hclosure Hc)
    as (slots' & Hc' & _).
  rewrite Hc'.
  assert (Hfull :
    known_arity_full statics' called args_arity
    = known_arity_full statics called args_arity).
  { apply (known_arity_full_mapped (bind_my_closure known called));
      [ apply (proj1 Hstatics) | ].
    intros c _. split; [ | apply bind_my_closure_params_arity ].
    destruct c as [params my_closure ret exn metadata body]. simpl.
    destruct (lookup called known); [ destruct (among my_closure params) | ];
      reflexivity. }
  rewrite Hfull.
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect_unknown_arity => true
     | Indirect_known_arity code_ids =>
       code_ids_allow code_ids called
       && known_arity_full statics called args_arity
     end);
    [ | discriminate ].
  rewrite (proj1 Hstatics called).
  destruct (lookup called (static_code statics)) as [c | ]; [ | discriminate ].
  simpl option_map. cbv iota zeta.
  assert (Htupled :
    code_is_tupled (bind_my_closure known called c) = code_is_tupled c).
  { destruct c as [params my_closure ret exn metadata body]. simpl.
    destruct (lookup called known); [ destruct (among my_closure params) | ];
      reflexivity. }
  destruct (arguments_for_code statics c vs args_arity)
    as [[ws args_arity'] | ] eqn:Hargs; [ | discriminate ].
  destruct
    (arguments_for_code_related (value_lifted statics') statics statics' c
       (bind_my_closure known called c) vs vs' args_arity ws args_arity'
       (value_lifted_respects statics statics' (proj2 Hstatics)) Htupled
       (bind_my_closure_params_arity called c) Hvs
       (kinds_of_values_lifted statics' vs vs' Hvs) Hargs)
    as (ws' & -> & Hws).
  simpl. unfold shape_of_code_call. cbv zeta.
  rewrite (bind_my_closure_params_arity called c),
    (bind_my_closure_result_arity called c),
    (kinds_of_values_lifted statics' ws ws' Hws), Htupled.
  clear Hargs Hvs vs vs'. rename ws into vs, ws' into vs', Hws into Hvs.
  rename args_arity into call_arity, args_arity' into args_arity.
  assert (Hcalled : called_lifted statics' closure closure' called).
  { split; [ exact Hclosure | ]. intros sym Hsym.
    exact (proj2
             (closure_of_free called sym statics' closure' called slots'
                (Hok called sym Hsym) (Hfree called sym Hsym) Hc')
             eq_refl). }
  destruct (Nat.eqb (length args_arity) (length (code_params_arity c))).
  - intros [= <-]. eexists. split; [ reflexivity | ].
    apply Full_call_lifted; assumption.
  - destruct
      (negb
         (kinds_eqb (map kind_of_value vs) (unarize args_arity)
          && partial_or_over_application args_arity return_arity
               (code_params_arity c) (code_result_arity c)));
      [ discriminate | ].
    destruct
      (Forall2_firstn_skipn (value_lifted statics')
         (cardinal_unarized (code_params_arity c)) vs vs' Hvs)
      as [Hfirst Hrest].
    destruct (Nat.ltb (length (code_params_arity c)) (length args_arity));
      [ intros [= <-]; eexists; (split; [ reflexivity | ]);
        apply Over_call_lifted; assumption
      | ].
    destruct (code_is_tupled c); [ discriminate | ].
    intros [= <-]. eexists. split; [ reflexivity | ].
    apply Partial_call_lifted, Lifted_partial_application; assumption.
Qed.

(* The same of a call: the closure that a partial application keeps is
   lifted with it. *)
Lemma shape_of_call_lifted (statics statics' : static_env)
    (callee callee' : value) (kind : call_kind) (vs vs' : list value)
    (args_arity return_arity : arity) (shape : call_shape) :
  statics_lifted statics statics' ->
  value_lifted statics' callee callee' ->
  Forall2 (value_lifted statics') vs vs' ->
  known_statics_ok statics' ->
  known_free callee' ->
  shape_of_call statics callee kind vs args_arity return_arity = Some shape ->
  ∃ shape' : call_shape,
    shape_of_call statics' callee' kind vs' args_arity return_arity
    = Some shape'
    ∧ shape_lifted statics' shape shape'.
Proof.
  intros Hstatics Hcallee Hvs Hok Hfree. unfold shape_of_call.
  destruct kind as [named | | code_ids];
    [ apply shape_of_closure_call_lifted; assumption |
    | apply shape_of_closure_call_lifted; assumption ].
  pose proof (partial_application_lifted statics' callee callee' Hcallee)
    as Hpartial.
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ].
  - destruct Hpartial as (closure' & applied' & -> & Hclosure & Happlied).
    simpl.
    destruct args_arity as [ | component args_arity]; [ discriminate | ].
    rewrite (kinds_of_values_lifted statics' vs vs' Hvs).
    destruct (kinds_eqb _ _); [ | discriminate ].
    apply shape_of_closure_call_lifted;
      [ exact Hstatics | exact Hclosure | apply Forall2_app; assumption
      | exact Hok | ].
    exact (proj1 (proj1 (known_free_partial _ _ _) Hfree)).
  - rewrite Hpartial. apply shape_of_closure_call_lifted; assumption.
Qed.

(* A full call. Where the code called has [my_closure] bound to a
   symbol, the second run takes one step more, to bind it; the closure
   called is then that symbol. *)
Lemma call_code_lifted (closure closure' : value) (called : code_id)
    (c : code) (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record) :
  called_lifted statics' closure closure' called ->
  Forall2 (value_lifted statics') vs vs' ->
  cont_lifted statics' return_to return_to' ->
  cont_lifted statics' exn_to exn_to' ->
  statics_lifted statics statics' ->
  heap_alike (value_lifted statics') hp hp' ->
  result_lifted statics'
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure' (bind_my_closure known called c) vs' args_arity
       return_arity return_to' exn_to' statics' hp' rd').
Proof.
  intros [Hclosure Hknown] Hvs Hreturn Hexn Hstatics Hheap.
  unfold result_lifted, call_code.
  destruct (bind_my_closure_code known called c)
    as (Hparams & Hmy_closure & Hret & Hexn_k & Hbody).
  rewrite Hparams, Hmy_closure, Hret, Hexn_k,
    (bind_my_closure_params_arity called c),
    (bind_my_closure_result_arity called c),
    (kinds_of_values_lifted statics' vs vs' Hvs).
  destruct (binds_statics (code_body c)) eqn:Hbinds; [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [bound | ] eqn:Hbind;
    [ | exact I ].
  destruct (bind_params_lifted statics' (code_params c) vs vs'
              [(code_my_closure c, closure)] [(code_my_closure c, closure')]
              bound Hvs)
    as (bound' & Hbind' & Hbound);
    [ apply env_lifted_cons; [ exact Hclosure | intros x v [=] ]
    | exact Hbind | ].
  assert (Hconts :
    conts_lifted statics'
      [(code_return_continuation c, return_to);
       (code_exn_continuation c, exn_to)]
      [(code_return_continuation c, return_to');
       (code_exn_continuation c, exn_to')]).
  { constructor; [ split; [ reflexivity | exact Hreturn ] | ].
    constructor; [ split; [ reflexivity | exact Hexn ] | ].
    constructor. }
  destruct Hbody as [Hbody | (sym & Hsym & Hparam & Hbody)]; rewrite Hbody.
  - (* [my_closure] is not bound in this code. *)
    rewrite bind_my_closures_binds_statics, Hbinds, Hbind'.
    eexists 0, _, _. split; [ reflexivity | ]. split; [ reflexivity | ].
    pose proof (State_lifted [] root (code_body c) bound bound' _ _ statics
      statics' hp hp' rd rd' Hbound Hconts Hstatics Hheap) as Hstate.
    rewrite lift_expr_nil in Hstate. exact Hstate.
  - (* It is: one step more, and the closure called is the symbol. *)
    simpl binds_statics.
    rewrite bind_my_closures_binds_statics, Hbinds, Hbind'.
    eexists 1, _,
      (State (bind_my_closures known (code_body c))
         ((code_my_closure c, Value_symbol sym) :: bound')
         [(code_return_continuation c, return_to');
          (code_exn_continuation c, exn_to')]
         statics' hp' rd').
    split; [ reflexivity | ].
    split; [ simpl; rewrite step_let_simple; reflexivity | ].
    pose proof (State_lifted [] root (code_body c) bound
      ((code_my_closure c, Value_symbol sym) :: bound')
      [(code_return_continuation c, return_to);
       (code_exn_continuation c, exn_to)]
      [(code_return_continuation c, return_to');
       (code_exn_continuation c, exn_to')]
      statics statics' hp hp' rd rd')
      as Hstate.
    rewrite lift_expr_nil in Hstate.
    apply Hstate; [ | exact Hconts | exact Hstatics | exact Hheap ].
    intros x v Hx. simpl.
    destruct (Pos.eqb_spec x (code_my_closure c)) as [-> | _];
      [ | exact (Hbound x v Hx) ].
    rewrite (bind_params_lookup_other (code_my_closure c) (code_params c) vs
               _ bound Hparam Hbind), lookup_head in Hx.
    injection Hx as <-. exists (Value_symbol sym). split; [ reflexivity | ].
    rewrite (Hknown sym Hsym) in Hclosure. exact Hclosure.
Qed.

(* A call of a given shape, given what the two runs do with a partial
   application. The full call inside an over-application returns to
   what is left of it. *)
Lemma run_call_lifted (return_partial return_partial' : value -> step_result)
    (shape shape' : call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record) :
  shape_lifted statics' shape shape' ->
  shape_values_all known_free (Some shape') ->
  cont_lifted statics' return_to return_to' ->
  cont_lifted statics' exn_to exn_to' ->
  statics_lifted statics statics' ->
  heap_alike (value_lifted statics') hp hp' ->
  (∀ partial partial' : value,
    value_lifted statics' partial partial' ->
    known_free partial' ->
    result_lifted statics' (return_partial partial)
      (return_partial' partial')) ->
  result_lifted statics'
    (run_call return_partial (Some shape) return_arity return_to exn_to
       statics hp rd)
    (run_call return_partial' (Some shape') return_arity return_to' exn_to'
       statics' hp' rd').
Proof.
  intros Hshape Hvalues Hreturn Hexn Hstatics Hheap Hpartial.
  destruct Hshape
    as [ closure closure' called c vs vs' args_arity Hcalled Hvs
       | closure closure' called c vs vs' remaining remaining' args_arity
           remaining_arity Hcalled Hvs Hremaining
       | partial partial' Hp ];
    simpl in Hvalues |- *.
  - apply call_code_lifted; assumption.
  - rewrite bind_my_closure_result_arity.
    apply call_code_lifted; try assumption.
    apply Over_application_lifted; assumption.
  - exact (Hpartial partial partial' Hp Hvalues).
Qed.

(* A call, given what the two runs do when the return continuation is
   applied to a partial application. *)
Lemma call_function_lifted_from (callee callee' : value) (kind : call_kind)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record) :
  value_lifted statics' callee callee' ->
  Forall2 (value_lifted statics') vs vs' ->
  cont_lifted statics' return_to return_to' ->
  cont_lifted statics' exn_to exn_to' ->
  statics_lifted statics statics' ->
  heap_alike (value_lifted statics') hp hp' ->
  known_statics_ok statics' ->
  known_free callee' ->
  Forall known_free vs' ->
  (∀ partial partial' : value,
    value_lifted statics' partial partial' ->
    known_free partial' ->
    result_lifted statics' (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial'] statics' hp' rd')) ->
  result_lifted statics'
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee' kind vs' args_arity return_arity return_to'
       exn_to' statics' hp' rd').
Proof.
  intros Hcallee Hvs Hreturn Hexn Hstatics Hheap Hok Hfree Hvs_free Hpartial.
  unfold call_function.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | exact I ].
  destruct (shape_of_call_lifted statics statics' callee callee' kind vs vs'
              args_arity return_arity shape Hstatics Hcallee Hvs Hok Hfree
              Hshape)
    as (shape' & Hshape' & Hshapes).
  assert (Hfields : P_fields known_free statics').
  { intros size tuple fields Htuple Htuple_fields.
    apply Forall_forall. intros v Hv id sym Hsym.
    pose proof
      (tuple_fields_free id sym statics' (Hok id sym Hsym) size tuple fields
         (Htuple id sym Hsym) Htuple_fields)
      as Hall.
    rewrite Forall_forall in Hall. exact (Hall v Hv). }
  pose proof
    (shape_of_call_values_all known_free known_free_partial statics' callee'
       kind vs' args_arity return_arity Hfields Hfree Hvs_free)
    as Hvalues.
  rewrite Hshape' in Hvalues |- *.
  exact (run_call_lifted _ _ shape shape' return_arity _ _ _ _ statics
           statics' _ _ _ _ Hshapes Hvalues Hreturn Hexn Hstatics Hheap
           Hpartial).
Qed.

(* Applying a continuation. What is left of an over-application calls
   the value it is given, which the second run holds free of the
   closures of [known]. *)
Lemma apply_cont_value_lifted (statics' : static_env) (c c' : cont_value) :
  cont_lifted statics' c c' ->
  ∀ (vs vs' : list value) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record),
    Forall2 (value_lifted statics') vs vs' ->
    statics_lifted statics statics' ->
    heap_alike (value_lifted statics') hp hp' ->
    known_statics_ok statics' ->
    known_cont_ok c' ->
    Forall known_free vs' ->
    result_lifted statics' (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs' statics' hp' rd').
Proof.
  induction 1
    as [ k | lets at_path [params metadata handler] vars vars' conts conts'
             recursive Hvars Hconts
       | c c' extra extra' Hc IH Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn IHreturn Hexn _ ];
    intros vs vs' statics hp hp' rd rd' Hvs Hstatics Hheap Hok Hcont Hfree.
  - simpl. exists vs', hp', rd'.
    split; [ reflexivity | split; [ exact Hvs | exact Hheap ] ].
  - simpl. rewrite (kinds_of_values_lifted statics' vs vs' Hvs).
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct (bind_params_lifted statics' params vs vs' vars vars' bound Hvs
                Hvars Hbind)
      as (bound' & -> & Hbound).
    eexists 0, _, _. split; [ reflexivity | ]. split; [ reflexivity | ].
    apply State_lifted;
      [ exact Hbound | apply rec_closures_lifted; assumption
      | exact Hstatics | exact Hheap ].
  - simpl. destruct (known_cont_ok_extra_args c' extra' Hcont)
      as [Hcont' Hextra_free].
    apply IH;
      [ apply Forall2_app; assumption | exact Hstatics | exact Hheap
      | exact Hok | exact Hcont' | apply Forall_app; split; assumption ].
  - rewrite !apply_cont_value_over_application.
    destruct Hvs as [ | v v' vs vs' Hv Hvs]; [ exact I | ].
    destruct Hvs as [ | w w' vs vs' Hw Hvs]; [ | exact I ].
    destruct (known_cont_ok_over_application _ _ _ _ _ Hcont)
      as [Hremaining_free Hreturn_ok].
    apply call_function_lifted_from; try assumption.
    + exact (Forall_inv Hfree).
    + intros partial partial' Hpartial Hpartial_free.
      apply IHreturn;
        [ constructor; [ exact Hpartial | constructor ] | exact Hstatics
        | exact Hheap | exact Hok | exact Hreturn_ok
        | constructor; [ exact Hpartial_free | constructor ] ].
Qed.

(* A call. *)
Lemma call_function_lifted (callee callee' : value) (kind : call_kind)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record) :
  value_lifted statics' callee callee' ->
  Forall2 (value_lifted statics') vs vs' ->
  cont_lifted statics' return_to return_to' ->
  cont_lifted statics' exn_to exn_to' ->
  statics_lifted statics statics' ->
  heap_alike (value_lifted statics') hp hp' ->
  known_statics_ok statics' ->
  known_free callee' ->
  Forall known_free vs' ->
  known_cont_ok return_to' ->
  result_lifted statics'
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee' kind vs' args_arity return_arity return_to'
       exn_to' statics' hp' rd').
Proof.
  intros Hcallee Hvs Hreturn Hexn Hstatics Hheap Hok Hfree Hvs_free
    Hreturn_ok.
  apply call_function_lifted_from; try assumption.
  intros partial partial' Hpartial Hpartial_free.
  apply (apply_cont_value_lifted statics' return_to return_to' Hreturn);
    [ constructor; [ exact Hpartial | constructor ] | exact Hstatics
    | exact Hheap | exact Hok | exact Hreturn_ok
    | constructor; [ exact Hpartial_free | constructor ] ].
Qed.

Lemma naked_immediate_lifted (statics' : static_env) (v v' : value) (i : Z) :
  value_lifted statics' v v' ->
  naked_immediate_of_value v = Some i ->
  naked_immediate_of_value v' = Some i.
Proof.
  intros Hv. lifted_cases Hv; simpl; try discriminate. exact (fun H => H).
Qed.

(* What a step of the second state answers. *)
Lemma result_lifted_answered (s' : state) (result : step_result) :
  result_lifted (state_statics s') result (step s') ->
  answered_heaps state_lifted exit_lifted result s'.
Proof.
  destruct result as [t | k vs hp rd | ]; unfold answered_heaps; simpl;
    [ | | exact (fun H => H) ].
  - intros (n & u' & t' & Hstep & Hrun & Ht). exists n, t'.
    split; [ rewrite Hstep; exact Hrun | exact Ht ].
  - intros (vs' & hp' & rd' & Hstep & Hvs & Hheap). exists vs', hp', rd'.
    split; [ exact Hstep | split; assumption ].
Qed.

(* A step of the first state is answered by the second, which is the
   state of a run that never binds a static name twice and that makes
   no closure of the code of [known] but its symbols: in one step, or in
   two where the first executes a [Let] that is lifted or calls code in
   which [my_closure] is bound. *)
Lemma lifted_step (s s' : state) :
  state_lifted s s' ->
  static_fault s' = false ->
  known_state_ok s' ->
  answered_heaps state_lifted exit_lifted (step s) s'.
Proof.
  intros [lets at_path e vars vars' conts conts' statics statics' hp hp' rd
            rd' Hvars Hconts Hstatics Hheap] Hfault Hknown.
  assert (Hok : known_statics_ok statics').
  { intros id sym Hsym. exact (proj2 (proj2 (proj2 (Hknown id sym Hsym)))). }
  assert (Hvars_ok :
    ∀ (id : code_id) (sym : symbol),
      lookup id known = Some sym -> env_free id vars').
  { intros id sym Hsym. exact (proj1 (proj2 (Hknown id sym Hsym))). }
  assert (Hconts_ok :
    ∀ (id : code_id) (sym : symbol),
      lookup id known = Some sym -> conts_ok id sym conts').
  { intros id sym Hsym. exact (proj1 (proj2 (proj2 (Hknown id sym Hsym)))). }
  clear Hknown.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - (* Let *)
    rewrite step_let_exec.
    destruct (exec_named statics hp rd vars d) as [[[v hp1] rd1] | ] eqn:Hd;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) xkind) eqn:Hkind; [ | exact I ].
    assert (Hkept :
      answered_heaps state_lifted exit_lifted
        (Step (State body ((x, v) :: vars) conts statics hp1 rd1))
        (State
           (bind_my_closures known
              (Let x xkind d (lift_expr lets (xO at_path) body)))
           vars' conts' statics' hp' rd')).
    { destruct (exec_named_lifted statics statics' hp hp' hp1 rd rd1 rd' vars
                  vars' d v (proj2 Hstatics) Hvars Hheap Hd)
        as (v' & hp1' & Hd' & Hv & Hheap1).
      exists 0, (State
                   (bind_my_closures known (lift_expr lets (xO at_path) body))
                   ((x, v') :: vars') conts' statics' hp1' rd').
      split;
        [ simpl;
          rewrite step_let_exec, Hd', (kind_of_value_lifted statics' v v' Hv),
            Hkind;
          reflexivity
        | ].
      apply State_lifted;
        [ apply env_lifted_cons; assumption | exact Hconts | exact Hstatics
        | exact Hheap1 ]. }
    simpl lift_expr in *.
    destruct (lookup at_path lets) as [target | ]; [ | exact Hkept ].
    destruct (lift_named target d) as [[bindings s0] | ] eqn:Hlift;
      [ | exact Hkept ].
    clear Hkept. simpl bind_my_closures in *.
    rewrite (lift_named_no_code target d bindings s0 Hlift) in *.
    destruct (exec_named_pure statics hp hp1 rd rd1 vars d v
                (lift_named_reads_no_heap target d bindings s0 Hlift) Hd)
      as (Heval & -> & ->).
    unfold static_fault in Hfault. simpl_state.
    apply orb_false_iff in Hfault as [_ Hfault].
    destruct (no_rebinding statics' bindings Hfault) as [Hdistinct Hfresh].
    destruct (lift_named_lifted target d bindings s0 statics statics' vars
                vars' v Hlift Hvars Heval Hdistinct Hfresh)
      as (extended' & Hextended & Hcode & Hv).
    pose proof (eval_static_bindings_extend vars' bindings statics' extended'
      Hextended) as Hextend.
    exists 1, (State
                 (bind_my_closures known (lift_expr lets (xO at_path) body))
                 ((x, Value_symbol s0) :: vars') conts' extended' hp' rd').
    split.
    + simpl. rewrite step_let_static, Hextended, step_let_simple.
      cbn [eval_simple].
      rewrite (kind_of_value_lifted _ _ _ Hv), Hkind. reflexivity.
    + apply State_lifted.
      * apply env_lifted_cons;
          [ exact Hv | exact (env_lifted_extend _ _ _ _ Hextend Hvars) ].
      * exact (conts_lifted_extend _ _ _ _ Hextend Hconts).
      * exact (statics_lifted_extend _ _ _ Hstatics Hextend Hcode).
      * exact (heap_lifted_extend _ _ _ _ Hextend Hheap).
  - (* Let_static *)
    simpl lift_expr in *. simpl bind_my_closures in *.
    rewrite step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings;
      [ | exact I ].
    unfold static_fault in Hfault. simpl_state.
    apply orb_false_iff in Hfault as [_ Hfault].
    destruct (no_rebinding statics' _ Hfault) as [Hdistinct Hfresh].
    rewrite bind_my_closures_symbols in Hdistinct, Hfresh.
    destruct (eval_static_bindings_lifted vars vars' bindings statics statics'
                extended Hvars Hstatics Hdistinct Hfresh Hbindings)
      as (extended' & Hextended & Hlifted).
    pose proof (eval_static_bindings_extend vars' _ statics' extended'
      Hextended) as Hextend.
    exists 0, (State
                 (bind_my_closures known (lift_expr lets (xO at_path) body))
                 vars' conts' extended' hp' rd').
    split; [ simpl; rewrite step_let_static, Hextended; reflexivity | ].
    apply State_lifted;
      [ exact (env_lifted_extend _ _ _ _ Hextend Hvars)
      | exact (conts_lifted_extend _ _ _ _ Hextend Hconts) | exact Hlifted
      | exact (heap_lifted_extend _ _ _ _ Hextend Hheap) ].
  - (* Let_cont_non_recursive *)
    simpl lift_expr. simpl bind_my_closures.
    rewrite step_let_cont_non_recursive.
    eexists 0, _.
    split; [ simpl; rewrite step_let_cont_non_recursive; reflexivity | ].
    apply State_lifted; [ exact Hvars | | exact Hstatics | exact Hheap ].
    constructor; [ | exact Hconts ].
    split; [ reflexivity | ].
    exact (Closure_lifted statics' lets (xI at_path) h vars vars' conts conts'
             [] Hvars Hconts).
  - (* Let_cont_recursive *)
    simpl lift_expr. simpl bind_my_closures.
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
    eexists 0, _.
    split;
      [ simpl;
        rewrite step_let_cont_recursive,
          bind_my_closures_handlers_bind_statics, Hbinds;
        reflexivity
      | ].
    apply State_lifted; [ exact Hvars | | exact Hstatics | exact Hheap ].
    apply rec_closures_lifted; assumption.
  - (* Apply *)
    simpl lift_expr in *. simpl bind_my_closures in *. rewrite step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra;
      [ | exact I ].
    destruct (eval_simple_lifted statics' vars vars' _ callee Hvars Hcallee)
      as (callee' & Hcallee' & Hlifted).
    destruct (eval_simples_lifted statics' vars vars' _ Hvars vs Hargs)
      as (vs' & Hargs' & Hvs).
    destruct (eval_simples_lifted statics' vars vars' _ Hvars extra Hextra)
      as (extra' & Hextra' & Hextras).
    apply result_lifted_answered. simpl state_statics.
    rewrite step_apply, Hcallee', Hargs', Hextra'.
    apply call_function_lifted; try assumption.
    + exact (resolve_lifted statics' _ conts conts' Hconts).
    + exact (with_extra_args_lifted statics' _ _ extra extra'
               (resolve_lifted statics' _ conts conts' Hconts) Hextras).
    + intros id sym Hsym.
      exact (eval_simple_free id vars' _ callee' (Hvars_ok id sym Hsym)
               Hcallee').
    + apply known_free_all. intros id sym Hsym.
      exact (eval_simples_free id vars' _ (Hvars_ok id sym Hsym) vs' Hargs').
    + intros id sym Hsym.
      exact (resolve_ok id sym _ conts' (Hconts_ok id sym Hsym)).
  - (* Apply_cont *)
    simpl lift_expr. simpl bind_my_closures. rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples_lifted statics' vars vars' _ Hvars vs Hargs)
      as (vs' & Hargs' & Hvs).
    apply result_lifted_answered. simpl state_statics.
    rewrite step_apply_cont, Hargs'. unfold apply_continuation.
    apply (apply_cont_value_lifted statics' _ _
             (resolve_lifted statics' _ conts conts' Hconts));
      [ exact Hvs | exact Hstatics | exact Hheap | exact Hok | | ].
    + intros id sym Hsym.
      exact (resolve_ok id sym _ conts' (Hconts_ok id sym Hsym)).
    + apply known_free_all. intros id sym Hsym.
      exact (eval_simples_free id vars' _ (Hvars_ok id sym Hsym) vs' Hargs').
  - (* Switch *)
    simpl lift_expr. simpl bind_my_closures. rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hscrutinee;
      [ | exact I ].
    destruct (eval_simple_lifted statics' vars vars' _ v Hvars Hscrutinee)
      as (v' & Hscrutinee' & Hv).
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    exists 0, (State (Apply_cont ac) vars' conts' statics' hp' rd').
    split.
    + simpl.
      rewrite step_switch, Hscrutinee',
        (naked_immediate_lifted statics' v v' i Hv Hi), Harm.
      reflexivity.
    + exact (State_lifted lets at_path (Apply_cont ac) vars vars' conts conts'
               statics statics' hp hp' rd rd' Hvars Hconts Hstatics Hheap).
  - (* Invalid *)
    rewrite step_invalid. exact I.
Qed.

(* ======================================================================== *)
(* The run of the unit lifted                                               *)
(* ======================================================================== *)

Section Unit.

Variable u : unit_interface.
Variable lets : list (path * lift_to).
Variable e : expr.

(* The static environment and the heap that the two runs start from:
   the environment holds no code, binds no name that the unit lifted
   binds, and binds nothing to a closure of the code of [known] but its
   symbol. The empty one is such, and so is that of the imports of a
   unit. No cell of the heap holds a closure of that code; the empty
   heap is such. *)
Variable statics0 : static_env.
Variable hp0 : heap.
Variable rd0 : read_record.

(* The continuation environment that the two runs start from: what may
   follow a unit (FlambdaObserversSpec.observers). The empty one is
   such. *)
Variable conts0 : cont_env.

(* The unit with its [Let]s lifted and [my_closure] bound in its code. *)
Definition lifted_unit : expr :=
  bind_my_closures known (lift_expr lets root e).

Hypothesis Hwf : well_formed_unit u lifted_unit = true.
Hypothesis Honly : only_closures_of known lifted_unit = true.
Hypothesis Hno_code : static_code statics0 = [].
Hypothesis Hfree : static_names_free_in statics0 lifted_unit = true.
Hypothesis Hstatics_ok :
  ∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> statics_ok id sym statics0.
Hypothesis Hheap_ok :
  ∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> heap_free id hp0.
Hypothesis Hobservers : observers [] conts0.

(* The second state is one that the run of the unit lifted reaches. *)
Definition reached (s' : state) : Prop :=
  ∃ n : nat, run n (State lifted_unit [] conts0 statics0 hp0 rd0) = Step s'.

(* The run of the unit lifted makes no closure of the code of [known]
   but its symbols. *)
Lemma reached_known_ok (s' : state) : reached s' -> known_state_ok s'.
Proof.
  intros [n Hn] id sym Hsym.
  unfold only_closures_of in Honly. rewrite forallb_forall in Honly.
  refine
    (ok_run id sym n (State lifted_unit [] conts0 statics0 hp0 rd0) s' _
       (Hheap_ok id sym Hsym) Hn).
  split; [ exact (Honly (id, sym) (lookup_in id sym known Hsym)) | ].
  split; [ intros x v [=] | ].
  split;
    [ exact (observers_closures_ok id sym [] conts0 Hobservers)
    | exact (Hstatics_ok id sym Hsym) ].
Qed.

Lemma reached_run (m : nat) (s' t' : state) :
  reached s' -> run m s' = Step t' -> reached t'.
Proof.
  intros [n Hn] Hm. exists (n + m). rewrite run_add, Hn. exact Hm.
Qed.

Definition unit_lifted (s s' : state) : Prop :=
  state_lifted s s' ∧ reached s'.

Lemma unit_lifted_step (s s' : state) :
  unit_lifted s s' -> answered_heaps unit_lifted exit_lifted (step s) s'.
Proof.
  intros [Hlifted Hreached].
  assert (Hfault : static_fault s' = false).
  { destruct Hreached as [n Hn].
    exact (well_formed_never_faults_under u [] _ conts0 statics0 hp0 rd0 n s'
             Hwf Hno_code Hfree Hobservers Hn). }
  pose proof (lifted_step s s' Hlifted Hfault
    (reached_known_ok s' Hreached)) as Hanswer.
  destruct (step s) as [t | k vs | ]; unfold answered_heaps in *;
    [ | exact Hanswer | exact I ].
  destruct Hanswer as (n & t' & Hrun & Ht).
  exists n, t'. split; [ exact Hrun | ].
  split; [ exact Ht | exact (reached_run (S n) s' t' Hreached Hrun) ].
Qed.

Lemma unit_lifted_initial :
  unit_lifted (State e [] conts0 statics0 hp0 rd0)
    (State lifted_unit [] conts0 statics0 hp0 rd0).
Proof.
  split; [ | exists 0; reflexivity ].
  apply State_lifted;
    [ intros x v [=] | exact (observers_lifted statics0 [] conts0 Hobservers)
    | split;
      [ intros id; rewrite Hno_code; reflexivity
      | apply symbols_lifted_refl ]
    | apply heap_lifted_refl ].
Qed.

(* The unit lifted, run from what may follow a unit, comes to what the
   unit comes to, up to what an exit is compared by. *)
Lemma unit_lifted_observed :
  observed_refines (Step (State e [] conts0 statics0 hp0 rd0))
    (Step (State lifted_unit [] conts0 statics0 hp0 rd0)).
Proof.
  apply (stuttering_heaps_observed unit_lifted exit_lifted);
    [ intros last vs hp vs' hp' [Hvs Hheap];
      exact (conj (lifted_same_scalars _ vs vs' Hvs)
               (heap_alike_keys _ hp hp' Hheap))
    | exact unit_lifted_step | exact unit_lifted_initial ].
Qed.

End Unit.

End Known.

(* ======================================================================== *)
(* What may follow a unit, after two exits that are lifted                  *)
(* ======================================================================== *)

Lemma nothing_known (P : code_id -> symbol -> Prop) (id : code_id)
    (sym : symbol) :
  lookup id ([] : list (code_id * symbol)) = Some sym -> P id sym.
Proof. intros [=]. Qed.

(* What may follow a unit, applied to the values of an exit and to the
   same values lifted, under static environments of which the second
   is the first lifted, with the same code, and on heaps alike: the
   two come to results that differ in nothing that an exit is compared
   by. It is the
   simulation of this file with no [Let] lifted and no code named, from
   the two states that the handler starts in. The second run is asked
   to make no static fault. *)
Lemma observer_applied_lifted (slots : list value_slot) (c : cont_value)
    (vs vs' : list value) (statics statics' : static_env) (hp hp' : heap)
      (rd rd' : read_record) :
  observer slots c ->
  Forall2 (value_lifted statics') vs vs' ->
  statics_lifted [] statics statics' ->
  heap_alike (value_lifted statics') hp hp' ->
  (∀ s' : state,
    apply_cont_value c vs' statics' hp' rd' = Step s' ->
    ∀ (m : nat) (t' : state), run m s' = Step t' -> static_fault t' = false) ->
  observed_refines (apply_cont_value c vs statics hp rd)
    (apply_cont_value c vs' statics' hp' rd').
Proof.
  intros Hobserver Hvs Hstatics Hheap Hfaults.
  assert (Hfree : Forall (known_free []) vs').
  { apply Forall_forall. intros v _. exact (nothing_known _). }
  pose proof
    (apply_cont_value_lifted [] statics' c c
       (observer_lifted [] statics' slots c Hobserver) vs vs' statics hp hp'
       rd rd' Hvs Hstatics Hheap (nothing_known _) (nothing_known _) Hfree)
    as Hresult.
  destruct (apply_cont_value c vs statics hp rd) as [t | k exit_vs hp1 rd1 | ];
    simpl in Hresult.
  - destruct Hresult as (n & u' & t' & Htarget & Hrun & Hlifted).
    rewrite Htarget.
    set (R2 :=
      fun s s' : state =>
        state_lifted [] s s'
        ∧ ∀ (m : nat) (t'' : state),
            run m s' = Step t'' -> static_fault t'' = false).
    assert (R2_step :
      ∀ s s' : state, R2 s s' -> answered_heaps R2 exit_lifted (step s) s').
    { intros s s' [Hl Hf].
      pose proof
        (lifted_step [] s s' Hl (Hf 0 s' eq_refl) (nothing_known _))
        as Hanswer.
      destruct (step s) as [t0 | k0 vs0 hp0 rd0 | ];
        unfold answered_heaps in *;
        [ | exact Hanswer | exact I ].
      destruct Hanswer as (j & t0' & Hrun0 & Hl0).
      exists j, t0'. split; [ exact Hrun0 | ]. split; [ exact Hl0 | ].
      intros m t'' Hm. apply (Hf (S j + m) t'').
      rewrite run_add, Hrun0. exact Hm. }
    assert (Hobserved : observed_refines (Step t) (Step t')).
    { apply (stuttering_heaps_observed R2 exit_lifted);
        [ intros last vs0 h0 vs0' h0' [H0 Hh0];
          exact (conj (lifted_same_scalars _ vs0 vs0' H0)
                   (heap_alike_keys _ h0 h0' Hh0))
        | exact R2_step | ].
      split; [ exact Hlifted | ].
      intros m t'' Hm. apply (Hfaults u' Htarget (n + m) t'').
      rewrite run_add, Hrun. exact Hm. }
    refine
      (observed_refines_ext (Step t) (Step t) (Step t') (Step u') _ _
         Hobserved).
    + intros b. reflexivity.
    + intros b. simpl. rewrite (run_behaves n u' b), Hrun. reflexivity.
  - destruct Hresult as (exit_vs' & hp1' & rd1' & Htarget & Hexit & Hheap1).
    rewrite Htarget. split.
    + intros k0 vs0 hp0 H. simpl in H. injection H as -> -> ->.
      exists exit_vs', hp1'.
      split; [ reflexivity | ].
      split;
        [ exact (lifted_same_scalars _ exit_vs exit_vs' Hexit)
        | exact (heap_alike_keys _ _ _ Hheap1) ].
    + intros b _ _. right. exists k, exit_vs, hp1. reflexivity.
  - split.
    + intros k0 vs0 hp0 H. simpl in H. discriminate H.
    + intros b _ _. left. reflexivity.
Qed.

(* The empty static environment is one that the two runs may start
   from, and so is that of the imports of a unit. *)
Lemma statics_ok_empty (id : code_id) (sym : symbol) :
  statics_ok id sym empty_static_env.
Proof. split; [ intros id' c [=] | intros s a [=] ]. Qed.

Lemma statics_ok_imports (id : code_id) (sym : symbol)
    (imports : list (symbol * import_class)) :
  statics_ok id sym (imports_statics imports).
Proof.
  split; [ intros id' c [=] | ].
  intros s allocated Hs.
  destruct (imports_statics_imported imports s allocated Hs) as [c ->].
  reflexivity.
Qed.

Theorem lift_simulates : lift_simulates_statement.
Proof.
  intros u lets known e. cbv zeta. intros Hwf Honly. split.
  - intros k vs hp [n [rd Hn]].
    destruct
      (stuttering_heaps_exit
         (unit_lifted known lets e empty_static_env empty_heap no_reads [])
         exit_lifted
         (unit_lifted_step known u lets e empty_static_env empty_heap
            no_reads [] Hwf Honly eq_refl (free_in_empty _)
            (fun id sym _ => statics_ok_empty id sym)
            (fun id sym _ => heap_values_all_empty _) (Forall_nil _))
         n _ _ k vs hp rd
         (unit_lifted_initial known lets e empty_static_env empty_heap
            no_reads [] eq_refl (Forall_nil _))
         Hn)
      as (m & last & vs' & hp' & rd' & Hm & Hlast & HQ & Hheap).
    exists m, last, vs', hp', rd'.
    exact (conj Hm (conj Hlast (conj HQ Hheap))).
  - intros b Hterm Hb.
    destruct (behaves_ended _ b Hterm Hb) as [m Hended].
    destruct (stuttering_heaps_ends
                (unit_lifted known lets e empty_static_env empty_heap no_reads
                   [])
                exit_lifted
                (unit_lifted_step known u lets e empty_static_env empty_heap
                   no_reads
                   [] Hwf Honly eq_refl (free_in_empty _)
                   (fun id sym _ => statics_ok_empty id sym)
                   (fun id sym _ => heap_values_all_empty _) (Forall_nil _))
                m _ _ b
                (unit_lifted_initial known lets e empty_static_env empty_heap
                   no_reads
                   [] eq_refl (Forall_nil _))
                Hended)
      as [n [Hwrong | (k & vs & hp & rd & Hexit)]].
    + left. exists n. exact Hwrong.
    + right. exists k, vs, hp, n, rd. exact Hexit.
Qed.

(* ======================================================================== *)
(* What a unit may exit with                                                *)
(* ======================================================================== *)

(* A continuation that is no way out, whatever it is applied to: a
   closure, or what is left of an over-application that returns to
   one. *)
Inductive cont_closed : cont_value -> Prop :=
  | Closure_closed (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    cont_closed (Closure h vars conts recursive)
  | Over_application_closed (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    cont_closed return_to ->
    cont_closed
      (Over_application remaining remaining_arity return_arity return_to
         exn_to).

(* A full call steps into code or is undefined, and so does a call
   where what is done with a partial application does. *)
Lemma call_code_no_exit (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (k : continuation)
    (exit_vs : list value) (hp' : heap) (rd' : read_record) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  <> Exit k exit_vs hp' rd'.
Proof.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [H | [callee_vars H]];
    rewrite H; discriminate.
Qed.

Lemma run_call_no_exit (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
    (k : continuation) (exit_vs : list value) (hp' : heap) (rd' : read_record) :
  (∀ partial : value, return_partial partial <> Exit k exit_vs hp' rd') ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  <> Exit k exit_vs hp' rd'.
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl;
    [ apply call_code_no_exit | apply call_code_no_exit | apply Hpartial
    | discriminate ].
Qed.

(* Applying a continuation that is no way out is no exit. *)
Lemma apply_cont_value_closed (c : cont_value) :
  cont_closed c ->
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (k : continuation)
    (exit_vs : list value) (hp' : heap) (rd' : read_record),
    apply_cont_value c vs statics hp rd <> Exit k exit_vs hp' rd'.
Proof.
  induction 1
    as [ [params metadata handler] vars conts recursive
       | remaining remaining_arity return_arity return_to exn_to _ IH ];
    intros vs statics hp rd k exit_vs hp' rd'.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs vars); discriminate.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    unfold call_function. apply run_call_no_exit. intros partial. apply IH.
Qed.

(* The continuations of [scope] are bound to continuations that are no
   way out. *)
Definition scope_closed (scope : list continuation) (conts : cont_env)
    : Prop :=
  ∀ k : continuation,
    among k scope = true ->
    ∃ c : cont_value, lookup k conts = Some c ∧ cont_closed c.

(* ------------------------------------------------------------------------ *)
(* What an expression spoils                                                *)
(* ------------------------------------------------------------------------ *)

Lemma spoils_at_app (k : continuation) (i : nat) (sp sp' : spoilt) :
  spoils_at k i (sp ++ sp') = spoils_at k i sp || spoils_at k i sp'.
Proof. apply existsb_app. Qed.

(* What is said of a continuation that is forgotten is said no longer,
   and what is said of another is kept. *)
Lemma spoils_at_forget (k : continuation) (i : nat) (ks : list continuation)
    (sp : spoilt) :
  spoils_at k i (forget_conts ks sp) = spoils_at k i sp && negb (among k ks).
Proof.
  induction sp as [ | [k' at_] sp IH]; simpl; [ reflexivity | ].
  destruct (among k' ks) eqn:Hk'; simpl; rewrite IH;
    destruct (Pos.eqb_spec k k') as [-> | _]; simpl; try reflexivity;
    rewrite Hk'; simpl.
  - rewrite !andb_false_r. reflexivity.
  - rewrite !andb_true_r. reflexivity.
Qed.

Lemma spoils_at_flat_map {A : Type} (f : A -> spoilt) (k : continuation)
    (i : nat) (l : list A) :
  spoils_at k i (flat_map f l) = existsb (fun a => spoils_at k i (f a)) l.
Proof.
  induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite spoils_at_app, IH. reflexivity.
Qed.

(* What the arguments of an application spoil is of the continuation
   applied. *)
Lemma args_spoil_other (k k' : continuation) (plain : list variable)
    (j : nat) (args : list simple) :
  k <> k' -> ∀ i : nat, spoils_at k j (args_spoil k' plain i args) = false.
Proof.
  intros Hne. induction args as [ | s args IH]; intros i; simpl;
    [ reflexivity | ].
  rewrite spoils_at_app, IH, orb_false_r.
  destruct (plain_or plain s); simpl; [ reflexivity | ].
  destruct (Pos.eqb_spec k k'); [ contradiction | reflexivity ].
Qed.

(* Arguments that are all known spoil nothing. *)
Lemma args_spoil_none (k : continuation) (plain : list variable)
    (args : list simple) :
  forallb (plain_or plain) args = true ->
  ∀ i : nat, args_spoil k plain i args = [].
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  intros Hargs i. apply andb_true_iff in Hargs as [Hs Hargs].
  rewrite Hs, (IH Hargs). reflexivity.
Qed.

(* The argument at a position that is not spoilt is known. *)
Lemma args_spoil_at (k : continuation) (plain : list variable)
    (args : list simple) :
  ∀ (i j : nat) (s : simple),
    nth_error args j = Some s ->
    spoils_at k (i + j) (args_spoil k plain i args) = false ->
    plain_or plain s = true.
Proof.
  induction args as [ | s' args IH]; intros i [ | j] s; simpl;
    try discriminate.
  - intros [= ->]. rewrite Nat.add_0_r.
    destruct (plain_or plain s); [ reflexivity | ]. simpl.
    rewrite Pos.eqb_refl, Nat.eqb_refl. discriminate.
  - intros Hs Hsp. rewrite spoils_at_app in Hsp.
    apply orb_false_iff in Hsp as [_ Hsp]. rewrite Nat.add_succ_r in Hsp.
    exact (IH (S i) j s Hs Hsp).
Qed.

Lemma apply_cont_spoils_none (scope : list continuation)
    (plain : list variable) (ac : apply_cont) (k : continuation) (i : nat) :
  apply_cont_exits_plain_knowing scope plain ac = true ->
  among k scope = false ->
  spoils_at k i (apply_cont_spoils plain ac) = false.
Proof.
  unfold apply_cont_exits_plain_knowing, apply_cont_spoils. intros Hac Hk.
  destruct (among (ac_continuation ac) scope) eqn:Hin; simpl in Hac.
  - apply args_spoil_other. intros ->. rewrite Hin in Hk. discriminate.
  - rewrite (args_spoil_none _ plain _ Hac). reflexivity.
Qed.

(* An expression that passes the test for a scope spoils no parameter
   of a continuation that is not of the scope: what it says of one that
   it binds itself is forgotten. So code spoils no parameter of its
   exception continuation, unless that one has the name of its return
   continuation. *)
Fixpoint knowing_spoils_none (scope : list continuation)
    (plain : list variable) (e : expr) (k : continuation) (i : nat)
    {struct e} :
  exits_plain_knowing scope plain e = true ->
  among k scope = false ->
  spoils_at k i (spoils plain e) = false
with handler_knowing_spoils_none (scope : list continuation)
    (plain : list variable) (h : continuation_handler) (k : continuation)
    (i : nat) {struct h} :
  handler_exits_plain_knowing scope plain h = true ->
  among k scope = false ->
  spoils_at k i (handler_spoils plain h) = false.
Proof.
  - destruct e
      as [ x xkind d body | bindings body
         | k0 [params metadata handler] body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; intros He Hk.
    + exact (knowing_spoils_none scope _ body k i He Hk).
    + apply andb_true_iff in He as [_ He].
      exact (knowing_spoils_none scope _ body k i He Hk).
    + apply andb_true_iff in He as [Hh Hbody].
      rewrite spoils_at_app, spoils_at_forget,
        (knowing_spoils_none scope _ handler k i Hh Hk), orb_false_l.
      destruct (Pos.eqb k k0) eqn:Hk0;
        [ rewrite among_cons, Hk0; apply andb_false_r | ].
      rewrite (knowing_spoils_none (k0 :: scope) plain body k i Hbody);
        [ reflexivity | ].
      rewrite among_cons, Hk0. exact Hk.
    + apply andb_true_iff in He as [Hhandlers Hbody].
      rewrite spoils_at_forget.
      destruct (among k (map fst handlers)) eqn:Hgroup;
        [ apply andb_false_r | ].
      assert (Hk' : among k (map fst handlers ++ scope) = false).
      { rewrite among_app, Hgroup. exact Hk. }
      rewrite spoils_at_app,
        (knowing_spoils_none _ plain body k i Hbody Hk'), orb_false_r,
        andb_true_r, spoils_at_flat_map.
      clear Hgroup Hbody. revert Hhandlers Hk'.
      generalize (map fst handlers ++ scope) as scope'. intros scope'.
      induction handlers as [ | [k' h] handlers IH]; simpl;
        [ reflexivity | ].
      intros Hhandlers Hk'.
      apply andb_true_iff in Hhandlers as [Hh Hhandlers].
      rewrite (handler_knowing_spoils_none scope' plain h k i Hh Hk'),
        (IH Hhandlers Hk').
      reflexivity.
    + apply andb_true_iff in He as [Hret Hexn].
      assert (Hne : ∀ k' : continuation,
        among k' scope = true -> Pos.eqb k k' = false).
      { intros k' Hk'. destruct (Pos.eqb_spec k k') as [-> | _];
          [ rewrite Hk' in Hk; discriminate | reflexivity ]. }
      unfold spoils_at. simpl. rewrite (Hne _ Hret). simpl.
      destruct (forallb (plain_or plain) (exn_extra_simples ap));
        [ reflexivity | ].
      rewrite orb_false_r in Hexn. simpl. rewrite (Hne _ Hexn). reflexivity.
    + exact (apply_cont_spoils_none scope plain ac k i He Hk).
    + rewrite spoils_at_flat_map.
      induction arms as [ | [discriminant ac] arms IH]; simpl in *;
        [ reflexivity | ].
      apply andb_true_iff in He as [Hac Harms].
      rewrite (apply_cont_spoils_none scope plain ac k i Hac Hk),
        (IH Harms).
      reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply knowing_spoils_none.
Qed.

(* ------------------------------------------------------------------------ *)
(* What is known of the variables                                           *)
(* ------------------------------------------------------------------------ *)

(* The variables of [plain] that are bound hold constants and
   symbols. *)
Definition vars_plain (plain : list variable) (vars : env) : Prop :=
  ∀ (x : variable) (v : value),
    among x plain = true -> lookup x vars = Some v -> plain_value v.

Lemma vars_plain_fewer (plain plain' : list variable) (vars : env) :
  (∀ x : variable, among x plain' = true -> among x plain = true) ->
  vars_plain plain vars -> vars_plain plain' vars.
Proof. intros Hfewer Hvars x v Hx. exact (Hvars x v (Hfewer x Hx)). Qed.

(* A variable bound again is forgotten. *)
Lemma vars_plain_forgotten (plain : list variable) (vars : env)
    (x : variable) (v : value) :
  vars_plain plain vars -> vars_plain (without [x] plain) ((x, v) :: vars).
Proof.
  intros Hvars y w Hy. rewrite among_without in Hy.
  apply andb_true_iff in Hy as [Hy Hother]. apply negb_true_iff in Hother.
  rewrite among_cons in Hother. simpl in Hother.
  rewrite orb_false_r in Hother. rewrite lookup_cons, Hother.
  exact (Hvars y w Hy).
Qed.

(* A variable bound to a constant or a symbol is known. *)
Lemma vars_plain_known (plain : list variable) (vars : env) (x : variable)
    (v : value) :
  plain_value v ->
  vars_plain plain vars -> vars_plain (x :: plain) ((x, v) :: vars).
Proof.
  intros Hv Hvars y w Hy. rewrite lookup_cons. rewrite among_cons in Hy.
  destruct (Pos.eqb y x); [ intros [= <-]; exact Hv | exact (Hvars y w Hy) ].
Qed.

(* The parameters of a handler bound to values that are constants and
   symbols at the positions that are not spoilt: what [params_known]
   says is known. The two go through the parameters in the same
   order. *)
Lemma bind_params_known (k : continuation) (sp : spoilt)
    (params : list variable) :
  ∀ (i : nat) (vs : list value) (vars bound : env) (known : list variable),
    bind_params params vs vars = Some bound ->
    vars_plain known vars ->
    (∀ (j : nat) (v : value),
       nth_error vs j = Some v ->
       spoils_at k (i + j) sp = false -> plain_value v) ->
    vars_plain (params_known k sp i params known) bound.
Proof.
  induction params as [ | x params IH];
    intros i [ | v vs] vars bound known; simpl; try discriminate.
  - intros [= <-] Hvars _. exact Hvars.
  - intros Hbind Hvars Hvs. apply (IH (S i) vs _ bound _ Hbind).
    + destruct (spoils_at k i sp) eqn:Hsp;
        [ apply vars_plain_forgotten, Hvars | ].
      apply vars_plain_known; [ | exact Hvars ].
      apply (Hvs 0 v eq_refl). rewrite Nat.add_0_r. exact Hsp.
    + intros j w Hw Hsp. apply (Hvs (S j) w Hw). rewrite Nat.add_succ_r.
      exact Hsp.
Qed.

(* The parameters of a handler bound to any values: they are
   forgotten. *)
Lemma bind_params_forgotten (params : list variable) :
  ∀ (vs : list value) (vars bound : env) (plain : list variable),
    bind_params params vs vars = Some bound ->
    vars_plain plain vars ->
    vars_plain (without params plain) bound.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars bound plain;
    simpl; try discriminate.
  - intros [= <-]. apply vars_plain_fewer. intros y Hy.
    rewrite among_without in Hy. apply andb_true_iff in Hy as [Hy _].
    exact Hy.
  - intros Hbind Hvars.
    apply (vars_plain_fewer (without params (without [x] plain)));
      [ | exact (IH vs _ bound _ Hbind
                   (vars_plain_forgotten plain vars x v Hvars)) ].
    intros y. rewrite !among_without, !among_cons. simpl.
    destruct (among y plain), (Pos.eqb y x), (among y params);
      simpl; congruence.
Qed.

(* A simple that is known evaluates to a constant or a symbol. *)
Lemma eval_simple_known (plain : list variable) (vars : env) (s : simple)
    (v : value) :
  vars_plain plain vars ->
  plain_or plain s = true -> eval_simple vars s = Some v -> plain_value v.
Proof.
  intros Hvars Hs. destruct s as [x | sym | c]; simpl in *.
  - exact (Hvars x v Hs).
  - intros [= <-]. exact I.
  - intros [= <-]. exact I.
Qed.

(* A value among those of some simples is that of the simple at its
   position. *)
Lemma eval_simples_nth (vars : env) (args : list simple) :
  ∀ (vs : list value) (j : nat) (v : value),
    eval_simples vars args = Some vs ->
    nth_error vs j = Some v ->
    ∃ s : simple, nth_error args j = Some s ∧ eval_simple vars s = Some v.
Proof.
  induction args as [ | s args IH]; intros vs j v; simpl.
  - intros [= <-]. destruct j; discriminate.
  - destruct (eval_simple vars s) as [w | ] eqn:Hw; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    intros [= <-]. destruct j as [ | j]; simpl.
    + intros [= <-]. exists s. split; [ reflexivity | exact Hw ].
    + exact (IH rest j v eq_refl).
Qed.

Lemma eval_simples_known (plain : list variable) (vars : env)
    (args : list simple) (vs : list value) :
  vars_plain plain vars ->
  forallb (plain_or plain) args = true ->
  eval_simples vars args = Some vs ->
  ∀ v : value, v ∈ vs -> plain_value v.
Proof.
  intros Hvars Hargs Heval v Hin. apply In_nth_error in Hin as [j Hj].
  destruct (eval_simples_nth vars args vs j v Heval Hj) as (s & Hs & Hv).
  rewrite forallb_forall in Hargs.
  exact (eval_simple_known plain vars s v Hvars
           (Hargs s (nth_error_In _ _ Hs)) Hv).
Qed.

(* ------------------------------------------------------------------------ *)
(* What a continuation may be applied to                                    *)
(* ------------------------------------------------------------------------ *)

(* The values at the positions that [m] does not excuse are constants
   and symbols. *)
Definition args_fit (m : nat -> bool) (vs : list value) : Prop :=
  ∀ (i : nat) (v : value),
    nth_error vs i = Some v -> m i = false -> plain_value v.

Lemma args_fit_plain (m : nat -> bool) (vs : list value) :
  (∀ v : value, v ∈ vs -> plain_value v) -> args_fit m vs.
Proof. intros Hvs i v Hv _. exact (Hvs v (nth_error_In _ _ Hv)). Qed.

Lemma args_fit_any (m : nat -> bool) (vs : list value) :
  (∀ i : nat, m i = true) -> args_fit m vs.
Proof. intros Hm i v _ Hi. rewrite Hm in Hi. discriminate. Qed.

Lemma args_fit_none (m : nat -> bool) (vs : list value) :
  (∀ i : nat, m i = false) ->
  args_fit m vs -> ∀ v : value, v ∈ vs -> plain_value v.
Proof.
  intros Hm Hfit v Hin. apply In_nth_error in Hin as [i Hi].
  exact (Hfit i v Hi (Hm i)).
Qed.

(* [cont_plain c m]: applying [c] to values that are constants and
   symbols at the positions that [m] does not excuse keeps the
   invariant below, and if it is a way out the unit exits with
   constants and symbols.
   - A way out excuses no position.
   - The closure of a non-recursive handler excuses the positions that
     the scope of the handler spoils ([sp], for the name [k] that the
     handler was bound to): its body passes the test with the other
     parameters known, under variables that hold what is known where
     the [Let_cont] stands, and each continuation of its environment
     excuses what the body spoils of it.
   - The closure of a recursive handler excuses every position: no
     parameter of it is known.
   - A continuation with extra arguments applies the one inside to
     more values: that one excuses every position, or the extra
     arguments are constants and symbols and no position is excused.
   - What is left of an over-application excuses every position, of
     the one value it is given; it returns to a continuation that is
     no way out and excuses every position, as the call it comes from
     passed the test. *)
#[local] Set Warnings "-register-all".
Inductive cont_plain : cont_value -> (nat -> bool) -> Prop :=
  | Outer_plain (k : continuation) : cont_plain (Outer k) (fun _ => false)
  | Closure_plain (scope : list continuation) (plain : list variable)
      (k : continuation) (sp : spoilt) (params : list variable)
      (metadata : handler_metadata) (handler : expr) (vars : env)
      (conts : cont_env) :
    exits_plain_knowing scope (params_known k sp 0 params plain) handler
    = true ->
    vars_plain plain vars ->
    scope_closed scope conts ->
    (∀ (k' : continuation) (c : cont_value),
       lookup k' conts = Some c ->
       ∃ m : nat -> bool,
         cont_plain c m
         ∧ ∀ i : nat,
             spoils_at k' i
               (spoils (params_known k sp 0 params plain) handler)
             = true ->
             m i = true) ->
    cont_plain (Closure (Handler params metadata handler) vars conts [])
      (fun i => spoils_at k i sp)
  | Rec_closure_plain (scope : list continuation) (plain : list variable)
      (k : continuation) (h : continuation_handler) (vars : env)
      (conts : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    (k, h) ∈ recursive ->
    forallb
      (fun '(_, h') =>
         handler_exits_plain_knowing (map fst recursive ++ scope) plain h')
      recursive
    = true ->
    vars_plain plain vars ->
    scope_closed scope conts ->
    (∀ (k' : continuation) (c : cont_value),
       lookup k' conts = Some c ->
       ∃ m : nat -> bool,
         cont_plain c m
         ∧ ∀ i : nat,
             spoils_at k' i
               (forget_conts (map fst recursive)
                  (flat_map (fun '(_, h') => handler_spoils plain h')
                     recursive))
             = true ->
             m i = true) ->
    cont_plain (Closure h vars conts recursive) (fun _ => true)
  | Extra_args_plain (c : cont_value) (extra : list value)
      (m_c m : nat -> bool) :
    cont_plain c m_c ->
    (∀ j : nat, m_c j = true)
    ∨ ((∀ i : nat, m i = false) ∧ ∀ v : value, v ∈ extra -> plain_value v) ->
    cont_plain (Extra_args c extra) m
  | Over_application_plain (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) (m_return m_exn : nat -> bool) :
    cont_plain return_to m_return ->
    (∀ i : nat, m_return i = true) ->
    cont_closed return_to ->
    cont_plain exn_to m_exn ->
    cont_plain
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (fun _ => true).

(* Each continuation bound excuses what is spoilt of its name. *)
Definition conts_fit (sp : spoilt) (conts : cont_env) : Prop :=
  ∀ (k : continuation) (c : cont_value),
    lookup k conts = Some c ->
    ∃ m : nat -> bool,
      cont_plain c m ∧ ∀ i : nat, spoils_at k i sp = true -> m i = true.

Lemma conts_fit_less (sp sp' : spoilt) (conts : cont_env) :
  (∀ (k : continuation) (i : nat),
     spoils_at k i sp' = true -> spoils_at k i sp = true) ->
  conts_fit sp conts -> conts_fit sp' conts.
Proof.
  intros Hless Hfit k c Hk. destruct (Hfit k c Hk) as (m & Hc & Hm).
  exists m. split; [ exact Hc | ]. intros i Hi. exact (Hm i (Hless k i Hi)).
Qed.

Definition code_plain (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c ->
    code_exits_plain_knowing c = true.

(* The invariant: the expression passes the test for a scope that the
   continuations in scope close and for variables that hold constants
   and symbols, and each continuation bound excuses what the expression
   spoils of it. *)
Definition plain_state (s : state) : Prop :=
  ∃ (scope : list continuation) (plain : list variable),
    exits_plain_knowing scope plain (state_expr s) = true
    ∧ vars_plain plain (state_vars s)
    ∧ scope_closed scope (state_conts s)
    ∧ conts_fit (spoils plain (state_expr s)) (state_conts s)
    ∧ code_plain (state_statics s).

Lemma lookup_none_among {A : Type} (name : positive)
    (bindings : list (positive * A)) :
  lookup name bindings = None -> among name (map fst bindings) = false.
Proof.
  intros Hnone. destruct (among name (map fst bindings)) eqn:Hin;
    [ | reflexivity ].
  apply among_In in Hin. destruct (lookup_none_not_in name bindings Hnone Hin).
Qed.

(* What a handler of a recursive group spoils of a continuation that is
   not of the group, the group does. *)
Lemma group_spoils (plain : list variable)
    (recursive : list (continuation * continuation_handler))
    (k : continuation) (h : continuation_handler) (k' : continuation)
    (i : nat) :
  (k, h) ∈ recursive ->
  spoils_at k' i (forget_conts (map fst recursive) (handler_spoils plain h))
  = true ->
  spoils_at k' i
    (forget_conts (map fst recursive)
       (flat_map (fun '(_, h') => handler_spoils plain h') recursive))
  = true.
Proof.
  intros Hin. rewrite !spoils_at_forget, spoils_at_flat_map. intros Hsp.
  apply andb_true_iff in Hsp as [Hsp ->]. rewrite andb_true_r.
  apply existsb_exists. exists (k, h). split; [ exact Hin | exact Hsp ].
Qed.

(* The closures of a recursive group whose handlers pass the test,
   added to an environment: the names of the group are closed, and the
   continuations bound excuse what [sp] spoils if those of the
   environment excuse what it spoils of the names that are not of the
   group. *)
Lemma rec_closures_plain (scope : list continuation) (plain : list variable)
    (vars : env) (conts : cont_env)
    (recursive : list (continuation * continuation_handler)) (sp : spoilt) :
  forallb
    (fun '(_, h) =>
       handler_exits_plain_knowing (map fst recursive ++ scope) plain h)
    recursive
  = true ->
  vars_plain plain vars ->
  scope_closed scope conts ->
  conts_fit
    (forget_conts (map fst recursive)
       (flat_map (fun '(_, h) => handler_spoils plain h) recursive))
    conts ->
  conts_fit (forget_conts (map fst recursive) sp) conts ->
  scope_closed (map fst recursive ++ scope)
    (rec_closures recursive vars conts ++ conts)
  ∧ conts_fit sp (rec_closures recursive vars conts ++ conts).
Proof.
  intros Hrecursive Hvars Hscope Hgroup Hsp. split.
  - intros k Hk. rewrite lookup_rec_closures_app.
    destruct (lookup k recursive) as [h | ] eqn:Hlookup.
    + eexists. split; [ reflexivity | apply Closure_closed ].
    + assert (Hout : among k (map fst recursive) = false)
        by exact (lookup_none_among k recursive Hlookup).
      rewrite among_app, Hout in Hk. exact (Hscope k Hk).
  - intros k c. rewrite lookup_rec_closures_app.
    destruct (lookup k recursive) as [h | ] eqn:Hlookup.
    + intros [= <-]. exists (fun _ => true). split; [ | reflexivity ].
      exact (Rec_closure_plain scope plain k h vars conts recursive
               (lookup_in k h recursive Hlookup) Hrecursive Hvars Hscope
               Hgroup).
    + intros Hc. destruct (Hsp k c Hc) as (m & Hm & Hfit).
      assert (Hout : among k (map fst recursive) = false)
        by exact (lookup_none_among k recursive Hlookup).
      exists m. split; [ exact Hm | ]. intros i Hi. apply Hfit.
      rewrite spoils_at_forget, Hi, Hout. reflexivity.
Qed.

Lemma eval_static_bindings_plain (vars : env) :
  ∀ (bindings : list static_binding) (statics extended : static_env),
    forallb binding_exits_plain_knowing bindings = true ->
    code_plain statics ->
    eval_static_bindings vars bindings statics = Some extended ->
    code_plain extended.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics extended;
    simpl.
  - intros _ Hcode [= <-]. exact Hcode.
  - intros Hbindings Hcode.
    apply andb_true_iff in Hbindings as [Hbinding Hbindings].
    destruct binding as [id c | id | sym const].
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      apply (IH _ _ Hbindings). intros id' c'. simpl.
      destruct (Pos.eqb id' id);
        [ intros [= <-]; exact Hbinding | apply Hcode ].
    + exact (IH _ _ Hbindings Hcode).
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const); [ | discriminate ].
      apply (IH _ _ Hbindings). exact Hcode.
Qed.

(* A result of a step that keeps the invariant and is no exit. *)
Definition stays_plain (result : step_result) : Prop :=
  match result with
  | Step t => plain_state t
  | Exit _ _ _ _ => False
  | Undefined => True
  end.

(* Applying a continuation that is no way out, where applying it keeps
   the invariant. *)
Lemma closed_stays_plain (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_closed c ->
  match apply_cont_value c vs statics hp rd with
  | Step t => plain_state t
  | Exit k exit_vs _ _ => ∀ v : value, v ∈ exit_vs -> plain_value v
  | Undefined => True
  end ->
  stays_plain (apply_cont_value c vs statics hp rd).
Proof.
  intros Hclosed. unfold stays_plain.
  generalize (apply_cont_value_closed c Hclosed vs statics hp rd).
  destruct (apply_cont_value c vs statics hp rd) as [t | k exit_vs hp' rd' | ];
    intros Hno_exit Hplain;
    [ exact Hplain | exact (Hno_exit k exit_vs hp' rd' eq_refl) | exact I ].
Qed.

(* A full call of code that passes the test, returning to a
   continuation that is no way out and excuses every position, keeps
   the invariant. The return continuation of the code is bound first:
   its name stands for what the call returns to, also where the
   exception continuation of the code has that name, the entry of the
   return continuation hiding the other. Under another name the
   exception continuation is not of the scope of the body, which
   spoils no parameter of it, whatever it excuses. *)
Lemma call_code_plain (closure : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (m_return m_exn : nat -> bool)
    (statics : static_env) (hp : heap) (rd : read_record) :
  lookup called (static_code statics) = Some c ->
  code_plain statics ->
  cont_plain return_to m_return ->
  (∀ i : nat, m_return i = true) ->
  cont_closed return_to ->
  cont_plain exn_to m_exn ->
  stays_plain
    (call_code closure c vs args_arity return_arity return_to exn_to statics
       hp rd).
Proof.
  intros Hlookup Hcode Hreturn Hany Hclosed Hexn. unfold stays_plain.
  destruct
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    as [t | k exit_vs hp' rd' | ] eqn:Hcall;
    [ | exact (call_code_no_exit _ _ _ _ _ _ _ _ _ rd_ _ _ _ rd_ Hcall) | exact
     I ].
  apply call_code_step in Hcall as (callee_vars & _ & _ & _ & _ & _ & ->).
  pose proof (Hcode called c Hlookup) as Hc.
  destruct c as [params my_closure ret exn metadata body]. simpl in Hc |- *.
  exists [ret], []. simpl.
  split; [ exact Hc | ]. split; [ intros x v Hx; discriminate Hx | ].
  split; [ | split; [ | exact Hcode ] ].
  - intros k' Hk'. rewrite among_cons in Hk'. simpl in Hk'.
    rewrite orb_false_r in Hk'. simpl. rewrite Hk'.
    exists return_to. split; [ reflexivity | exact Hclosed ].
  - intros k' c'. simpl. destruct (Pos.eqb k' ret) eqn:Hret.
    + intros [= <-]. exists m_return.
      split; [ exact Hreturn | intros i _; apply Hany ].
    + destruct (Pos.eqb k' exn); [ | discriminate ].
      intros [= <-]. exists m_exn. split; [ exact Hexn | ]. intros i Hi.
      rewrite (knowing_spoils_none [ret] [] body k' i Hc) in Hi;
        [ discriminate | ].
      rewrite among_cons, Hret. reflexivity.
Qed.

(* So does a call of a given shape, where what is done with a partial
   application does. The full call inside an over-application returns
   to what is left of it, which is no way out. *)
Lemma run_call_plain (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (m_return m_exn : nat -> bool)
    (statics : static_env) (hp : heap) (rd : read_record) :
  shape_calls statics kind shape ->
  code_plain statics ->
  cont_plain return_to m_return ->
  (∀ i : nat, m_return i = true) ->
  cont_closed return_to ->
  cont_plain exn_to m_exn ->
  (∀ partial : value, stays_plain (return_partial partial)) ->
  stays_plain
    (run_call return_partial shape return_arity return_to exn_to statics hp rd).
Proof.
  intros Hcalls Hcode Hreturn Hany Hclosed Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    exact (call_code_plain closure called c vs _ _ _ _ _ _ statics _ rd_ Hlookup
             Hcode Hreturn Hany Hclosed Hexn).
  - destruct Hcalls as (called & slots & _ & Hlookup & _).
    apply
      (call_code_plain closure called c vs _ _ _ _ (fun _ => true) m_exn
         statics _ rd_ Hlookup Hcode);
      [ exact (Over_application_plain _ _ _ _ _ _ _ Hreturn Hany Hclosed Hexn)
      | reflexivity | apply Over_application_closed; exact Hclosed
      | exact Hexn ].
  - apply Hpartial.
  - exact I.
Qed.

(* The exception continuation that a call gives its callee: the
   caller's excuses every position, or the extra arguments are
   constants and symbols. *)
Lemma with_extra_args_plain (c : cont_value) (extra : list value)
    (m_c : nat -> bool) :
  cont_plain c m_c ->
  (∀ j : nat, m_c j = true) ∨ (∀ v : value, v ∈ extra -> plain_value v) ->
  ∃ m : nat -> bool, cont_plain (with_extra_args c extra) m.
Proof.
  intros Hc Hextra. destruct extra as [ | v extra]; [ exists m_c; exact Hc | ].
  exists (fun _ => false). apply (Extra_args_plain c _ m_c); [ exact Hc | ].
  destruct Hextra as [Hany | Hextra]; [ left; exact Hany | right ].
  split; [ reflexivity | exact Hextra ].
Qed.

(* Applying a continuation to values that are constants and symbols at
   the positions that it does not excuse keeps the invariant, and if it
   is a way out the values are what the term exits with. *)
Lemma apply_cont_value_plain (c : cont_value) (m : nat -> bool) :
  cont_plain c m ->
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    args_fit m vs ->
    code_plain statics ->
    match apply_cont_value c vs statics hp rd with
    | Step t => plain_state t
    | Exit k exit_vs _ _ => ∀ v : value, v ∈ exit_vs -> plain_value v
    | Undefined => True
    end.
Proof.
  induction 1
    as [ k
       | scope plain k sp params metadata handler dvars dconts Hh Hdvars
           Hscope Hdconts
       | scope plain k [params metadata handler] dvars dconts recursive Hin
           Hrecursive Hdvars Hscope Hdconts
       | c extra m_c m Hc IH Hextra
       | remaining remaining_arity return_arity return_to exn_to m_return
           m_exn Hreturn IHreturn Hany Hclosed Hexn _ ];
    intros vs statics hp rd Hvs Hcode.
  - simpl. exact (args_fit_none _ vs (fun _ => eq_refl) Hvs).
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs dvars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    exists scope, (params_known k sp 0 params plain). simpl.
    split; [ exact Hh | ].
    split;
      [ | split; [ exact Hscope | split; [ exact Hdconts | exact Hcode ] ] ].
    exact (bind_params_known k sp params 0 vs dvars bound plain Hbind Hdvars
             Hvs).
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs dvars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct
      (rec_closures_plain scope plain dvars dconts recursive
         (spoils (without params plain) handler) Hrecursive Hdvars Hscope
         Hdconts)
      as [Hscope' Hconts'].
    { refine (conts_fit_less _ _ dconts _ Hdconts). intros k' i.
      exact (group_spoils plain recursive k
               (Handler params metadata handler) k' i Hin). }
    exists (map fst recursive ++ scope), (without params plain). simpl.
    split.
    { rewrite forallb_forall in Hrecursive. exact (Hrecursive _ Hin). }
    split; [ exact (bind_params_forgotten params vs dvars bound plain Hbind
                      Hdvars) | ].
    auto.
  - simpl. apply IH; [ | exact Hcode ].
    destruct Hextra as [Hany | [Hnone Hextra]];
      [ apply args_fit_any, Hany | ].
    apply args_fit_plain. intros v Hin. apply in_app_or in Hin.
    destruct Hin as [Hin | Hin];
      [ exact (args_fit_none m vs Hnone Hvs v Hin) | exact (Hextra v Hin) ].
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    unfold call_function.
    generalize
      (run_call_plain
         (fun partial => apply_cont_value return_to [partial] statics hp rd)
         Indirect_unknown_arity _ return_arity return_to exn_to m_return m_exn
           statics hp rd
         (shape_of_call_calls statics v Indirect_unknown_arity remaining
            remaining_arity return_arity)
         Hcode Hreturn Hany Hclosed Hexn
         (fun partial =>
            closed_stays_plain return_to [partial] statics _ rd_ Hclosed
              (IHreturn [partial] statics hp rd_
                 (args_fit_any m_return [partial] Hany) Hcode))).
    unfold stays_plain.
    destruct (run_call _ _ _ _ _ _ _ rd_) as [t | k exit_vs | ];
      [ exact (fun H => H) | intros [] | exact (fun _ => I) ].
Qed.

(* A step keeps the invariant, and an exit is with constants and symbols
   only: the continuation left through is not of the scope, so it was
   given what is known only. *)
Lemma plain_step_components (scope : list continuation)
    (plain : list variable) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  exits_plain_knowing scope plain e = true ->
  vars_plain plain vars ->
  scope_closed scope conts ->
  conts_fit (spoils plain e) conts ->
  code_plain statics ->
  match step (State e vars conts statics hp rd) with
  | Step t => plain_state t
  | Exit k vs _ _ => ∀ v : value, v ∈ vs -> plain_value v
  | Undefined => True
  end.
Proof.
  intros He Hvars Hscope Hconts Hcode.
  destruct e
    as [ x xkind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl in He, Hconts.
  - (* Let *)
    rewrite step_let_exec.
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
    destruct (kind_eqb (kind_of_value v) xkind); [ | exact I ].
    exists scope, (without [x] plain). simpl.
    split; [ exact He | ].
    split; [ exact (vars_plain_forgotten plain vars x v Hvars) | auto ].
  - (* Let_static *)
    apply andb_true_iff in He as [Hbindings Hbody]. rewrite step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hextended;
      [ | exact I ].
    exists scope, plain. simpl.
    split; [ exact Hbody | ]. split; [ exact Hvars | ].
    split; [ exact Hscope | ]. split; [ exact Hconts | ].
    exact (eval_static_bindings_plain vars bindings statics extended Hbindings
             Hcode Hextended).
  - (* Let_cont_non_recursive *)
    apply andb_true_iff in He as [Hh Hbody].
    rewrite step_let_cont_non_recursive.
    exists (k :: scope), plain. simpl.
    split; [ exact Hbody | ]. split; [ exact Hvars | ].
    split; [ | split; [ | exact Hcode ] ].
    + intros k' Hk'. rewrite among_cons in Hk'. simpl.
      destruct (Pos.eqb k' k); [ | exact (Hscope k' Hk') ].
      eexists. split; [ reflexivity | apply Closure_closed ].
    + intros k' c. simpl. destruct (Pos.eqb_spec k' k) as [-> | Hne].
      * intros [= <-]. exists (fun i => spoils_at k i (spoils plain body)).
        split; [ | intros i Hi; exact Hi ].
        apply (Closure_plain scope plain k (spoils plain body));
          [ exact Hh | exact Hvars | exact Hscope | ].
        refine (conts_fit_less _ _ conts _ Hconts). intros k' i Hi.
        rewrite spoils_at_app, Hi. reflexivity.
      * intros Hc. destruct (Hconts k' c Hc) as (m & Hm & Hfit).
        exists m. split; [ exact Hm | ]. intros i Hi. apply Hfit.
        rewrite spoils_at_app, spoils_at_forget, Hi, among_cons.
        destruct (Pos.eqb_spec k' k); [ contradiction | ].
        apply orb_true_r.
  - (* Let_cont_recursive *)
    apply andb_true_iff in He as [Hhandlers Hbody].
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    destruct
      (rec_closures_plain scope plain vars conts handlers (spoils plain body)
         Hhandlers Hvars Hscope)
      as [Hscope' Hconts'].
    { refine (conts_fit_less _ _ conts _ Hconts). intros k i.
      rewrite !spoils_at_forget, spoils_at_app. intros Hi.
      apply andb_true_iff in Hi as [-> ->]. reflexivity. }
    { refine (conts_fit_less _ _ conts _ Hconts). intros k i.
      rewrite !spoils_at_forget, spoils_at_app. intros Hi.
      apply andb_true_iff in Hi as [-> ->]. rewrite orb_true_r.
      reflexivity. }
    exists (map fst handlers ++ scope), plain. simpl. auto.
  - (* Apply *)
    apply andb_true_iff in He as [Hret Hexn]. rewrite step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra;
      [ | exact I ].
    destruct (Hscope _ Hret) as (return_to & Hk & Hclosed).
    assert (Hresolve : resolve (ap_continuation ap) conts = return_to).
    { unfold resolve. rewrite Hk. reflexivity. }
    destruct (Hconts _ _ Hk) as (m_return & Hreturn & Hfit).
    assert (Hany : ∀ i : nat, m_return i = true).
    { intros i. apply Hfit. simpl. rewrite Pos.eqb_refl. reflexivity. }
    assert (Hexn_plain :
      ∃ m_exn : nat -> bool,
        cont_plain
          (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
          m_exn).
    { destruct (forallb (plain_or plain) (exn_extra_simples ap)) eqn:Hall.
      - assert (Hc :
          ∃ m_c : nat -> bool,
            cont_plain (resolve (ap_exn_continuation ap) conts) m_c).
        { unfold resolve.
          destruct (lookup (ap_exn_continuation ap) conts) as [c | ] eqn:Hc.
          - destruct (Hconts _ _ Hc) as (m_c & Hm_c & _).
            exists m_c. exact Hm_c.
          - eexists. apply Outer_plain. }
        destruct Hc as [m_c Hc]. apply (with_extra_args_plain _ _ m_c Hc).
        right. exact (eval_simples_known plain vars _ extra Hvars Hall Hextra).
      - rewrite orb_false_r in Hexn.
        destruct (Hscope _ Hexn) as (c & Hc & _).
        destruct (Hconts _ _ Hc) as (m_c & Hm_c & Hfit_c).
        unfold resolve. rewrite Hc.
        apply (with_extra_args_plain _ _ m_c Hm_c). left. intros j.
        apply Hfit_c. simpl. rewrite Pos.eqb_refl. apply orb_true_r. }
    destruct Hexn_plain as [m_exn Hexn_plain].
    rewrite Hresolve. unfold call_function.
    generalize
      (run_call_plain
         (fun partial => apply_cont_value return_to [partial] statics hp rd)
         (ap_call_kind ap) _ (ap_return_arity ap) return_to
         (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
         m_return m_exn statics hp rd
         (shape_of_call_calls statics callee (ap_call_kind ap) vs
            (ap_args_arity ap) (ap_return_arity ap))
         Hcode Hreturn Hany Hclosed Hexn_plain
         (fun partial =>
            closed_stays_plain return_to [partial] statics _ rd_ Hclosed
              (apply_cont_value_plain return_to m_return Hreturn [partial]
                 statics hp rd_ (args_fit_any m_return [partial] Hany)
                 Hcode))).
    unfold stays_plain.
    destruct (run_call _ _ _ _ _ _ _ rd_) as [t | k exit_vs | ];
      [ exact (fun H => H) | intros [] | exact (fun _ => I) ].
  - (* Apply_cont *)
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    unfold apply_continuation, resolve.
    destruct (lookup (ac_continuation ac) conts) as [c | ] eqn:Hk.
    + destruct (Hconts _ _ Hk) as (m & Hc & Hm).
      apply (apply_cont_value_plain c m Hc); [ | exact Hcode ].
      intros i v Hv Hi.
      destruct (eval_simples_nth vars _ vs i v Hargs Hv) as (s & Hs & Heval).
      apply (eval_simple_known plain vars s v Hvars); [ | exact Heval ].
      apply (args_spoil_at (ac_continuation ac) plain (ac_args ac) 0 i s Hs).
      simpl.
      destruct
        (spoils_at (ac_continuation ac) i
           (args_spoil (ac_continuation ac) plain 0 (ac_args ac)))
        eqn:Hsp;
        [ | reflexivity ].
      rewrite (Hm i Hsp) in Hi. discriminate.
    + apply (apply_cont_value_plain _ _ (Outer_plain (ac_continuation ac)));
        [ | exact Hcode ].
      apply args_fit_plain. unfold apply_cont_exits_plain_knowing in He.
      destruct (among (ac_continuation ac) scope) eqn:Hin.
      * destruct (Hscope _ Hin) as (c & Hc & _). rewrite Hk in Hc.
        discriminate.
      * exact (eval_simples_known plain vars _ vs Hvars He Hargs).
  - (* Switch *)
    rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    pose proof (find_arm_in i arms ac Harm) as Hin.
    exists scope, plain. simpl.
    split; [ rewrite forallb_forall in He; exact (He (i, ac) Hin) | ].
    split; [ exact Hvars | ]. split; [ exact Hscope | ].
    split; [ | exact Hcode ].
    refine (conts_fit_less _ _ conts _ Hconts). intros k j Hj.
    rewrite spoils_at_flat_map. apply existsb_exists.
    exists (i, ac). split; [ exact Hin | exact Hj ].
  - (* Invalid *)
    rewrite step_invalid. exact I.
Qed.

Lemma plain_step (s : state) :
  plain_state s ->
  match step s with
  | Step t => plain_state t
  | Exit k vs _ _ => ∀ v : value, v ∈ vs -> plain_value v
  | Undefined => True
  end.
Proof.
  intros (scope & plain & He & Hvars & Hscope & Hconts & Hcode).
  rewrite (state_eta s).
  exact (plain_step_components scope plain _ _ _ _ _ rd_ He Hvars Hscope Hconts
           Hcode).
Qed.

Lemma plain_run (n : nat) :
  ∀ (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    plain_state s ->
    run n s = Exit k vs hp rd ->
    ∀ v : value, v ∈ vs -> plain_value v.
Proof.
  induction n as [ | n IH]; intros s k vs hp rd Hplain; simpl;
    [ discriminate | ].
  pose proof (plain_step s Hplain) as Hstep.
  destruct (step s) as [t | k' vs' | ];
    [ exact (IH t k vs hp rd_ Hstep) | intros [= <- <-]; exact Hstep
    | discriminate ].
Qed.

(* The invariant holds of a first state whose static environment holds
   code that passes the test. *)
Lemma plain_from_code (e : expr) (statics : static_env) (hp : heap)
    (rd : read_record) :
  exits_plain_knowing [] [] e = true ->
  code_plain statics ->
  plain_state (State e [] [] statics hp rd).
Proof.
  intros He Hcode. exists [], []. simpl.
  split; [ exact He | ]. split; [ intros x v Hx; discriminate Hx | ].
  split; [ intros k Hk; discriminate Hk | ].
  split; [ intros k c Hk; discriminate Hk | exact Hcode ].
Qed.

(* So of one whose static environment holds no code. *)
Lemma plain_from (e : expr) (statics : static_env) (hp : heap)
    (rd : read_record) :
  exits_plain_knowing [] [] e = true ->
  static_code statics = [] ->
  plain_state (State e [] [] statics hp rd).
Proof.
  intros He Hno_code. apply (plain_from_code e statics hp rd He).
  intros id c Hlookup. rewrite Hno_code in Hlookup. discriminate Hlookup.
Qed.

Theorem exits_plain_knowing_sound_from :
  exits_plain_knowing_sound_from_statement.
Proof.
  intros statics e n k vs hp rd Hcode He.
  exact
    (plain_run n _ k vs _ rd_
       (plain_from_code e statics empty_heap no_reads He Hcode)).
Qed.

Theorem exits_plain_knowing_sound : exits_plain_knowing_sound_statement.
Proof.
  intros e n k vs hp rd He.
  exact (plain_run n _ k vs _ rd_
           (plain_from e empty_static_env empty_heap no_reads He eq_refl)).
Qed.

(* ------------------------------------------------------------------------ *)
(* The test without what is known                                           *)
(* ------------------------------------------------------------------------ *)

Lemma plain_simples_known (plain : list variable) (args : list simple) :
  forallb plain_simple args = true -> forallb (plain_or plain) args = true.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  intros Hargs. apply andb_true_iff in Hargs as [Hs Hargs].
  rewrite (IH Hargs), andb_true_r.
  destruct s; [ discriminate | reflexivity | reflexivity ].
Qed.

Lemma apply_cont_exits_plain_accepted (scope : list continuation)
    (plain : list variable) (ac : apply_cont) :
  apply_cont_exits_plain scope ac = true ->
  apply_cont_exits_plain_knowing scope plain ac = true.
Proof.
  unfold apply_cont_exits_plain, apply_cont_exits_plain_knowing.
  destruct (among (ac_continuation ac) scope); [ reflexivity | ].
  apply plain_simples_known.
Qed.

Fixpoint exits_plain_accepted (scope : list continuation)
    (plain : list variable) (e : expr) {struct e} :
  exits_plain scope e = true -> exits_plain_knowing scope plain e = true
with handler_exits_plain_accepted (scope : list continuation)
    (plain : list variable) (h : continuation_handler) {struct h} :
  handler_exits_plain scope h = true ->
  handler_exits_plain_knowing scope plain h = true
with binding_exits_plain_accepted (binding : static_binding)
    {struct binding} :
  binding_exits_plain binding = true ->
  binding_exits_plain_knowing binding = true
with code_exits_plain_accepted (c : code) {struct c} :
  code_exits_plain c = true -> code_exits_plain_knowing c = true.
Proof.
  - destruct e
      as [ x xkind d body | bindings body
         | k [params metadata handler] body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; intros He.
    + exact (exits_plain_accepted scope _ body He).
    + apply andb_true_iff in He as [Hbindings Hbody].
      rewrite (exits_plain_accepted scope plain body Hbody), andb_true_r.
      induction bindings as [ | binding bindings IH]; simpl in *;
        [ reflexivity | ].
      apply andb_true_iff in Hbindings as [Hbinding Hbindings].
      rewrite (binding_exits_plain_accepted binding Hbinding),
        (IH Hbindings).
      reflexivity.
    + apply andb_true_iff in He as [Hh Hbody].
      rewrite (exits_plain_accepted scope _ handler Hh),
        (exits_plain_accepted (k :: scope) plain body Hbody).
      reflexivity.
    + apply andb_true_iff in He as [Hhandlers Hbody].
      rewrite (exits_plain_accepted _ plain body Hbody), andb_true_r.
      clear Hbody. revert Hhandlers.
      generalize (map fst handlers ++ scope) as scope'. intros scope'.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      intros Hhandlers. apply andb_true_iff in Hhandlers as [Hh Hhandlers].
      rewrite (handler_exits_plain_accepted scope' plain h Hh),
        (IH Hhandlers).
      reflexivity.
    + apply andb_true_iff in He as [-> Hexn]. simpl.
      destruct (among (ap_exn_continuation ap) scope); [ reflexivity | ].
      exact (plain_simples_known plain _ Hexn).
    + exact (apply_cont_exits_plain_accepted scope plain ac He).
    + induction arms as [ | [discriminant ac] arms IH]; simpl in *;
        [ reflexivity | ].
      apply andb_true_iff in He as [Hac Harms].
      rewrite (apply_cont_exits_plain_accepted scope plain ac Hac),
        (IH Harms).
      reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply exits_plain_accepted.
  - destruct binding as [id c | id | sym const]; simpl; try reflexivity.
    apply code_exits_plain_accepted.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    apply exits_plain_accepted.
Qed.

Theorem exits_plain_knowing_accepts : exits_plain_knowing_accepts_statement.
Proof. exact exits_plain_accepted. Qed.

(* So a unit that passes the test without what is known exits with
   constants and symbols only. *)
Theorem exits_plain_sound : exits_plain_sound_statement.
Proof.
  intros e n k vs hp rd He.
  exact (exits_plain_knowing_sound e n k vs hp rd
           (exits_plain_accepted [] [] e He)).
Qed.

(* ======================================================================== *)
(* What [lift_unit] gives refines the unit                                  *)
(* ======================================================================== *)

(* What two exits are compared by where the first run exits with
   constants and symbols only: the same values, and heaps alike. *)
Definition exit_plain (last : state) (vs : list value) (hp : heap)
    (vs' : list value) (hp' : heap) : Prop :=
  vs' = vs ∧ heap_alike (value_lifted (state_statics last)) hp hp'.

Section Unit_refines.

Variable known : list (code_id * symbol).
Variable u : unit_interface.
Variable lets : list (path * lift_to).
Variable e : expr.
Variable statics0 : static_env.
Variable hp0 : heap.
Variable rd0 : read_record.
Hypothesis Hwf :
  well_formed_unit u (lifted_unit known lets e) = true.
Hypothesis Honly : only_closures_of known (lifted_unit known lets e) = true.
Hypothesis Hno_code : static_code statics0 = [].
Hypothesis Hfree :
  static_names_free_in statics0 (lifted_unit known lets e) = true.
Hypothesis Hstatics_ok :
  ∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> statics_ok id sym statics0.
Hypothesis Hheap_ok :
  ∀ (id : code_id) (sym : symbol),
    lookup id known = Some sym -> heap_free id hp0.

Definition unit_lifted_plain (s s' : state) : Prop :=
  unit_lifted known lets e statics0 hp0 rd0 [] s s' ∧ plain_state s.

(* The values at an exit are lifted, and are constants and symbols: they
   are the same. *)
Lemma unit_lifted_plain_step (s s' : state) :
  unit_lifted_plain s s' ->
  answered_heaps unit_lifted_plain exit_plain (step s) s'.
Proof.
  intros [Hlifted Hplain].
  pose proof
    (unit_lifted_step known u lets e statics0 hp0 rd0 [] Hwf Honly Hno_code
       Hfree
       Hstatics_ok Hheap_ok (Forall_nil _) s s' Hlifted)
    as Hanswer.
  pose proof (plain_step s Hplain) as Hstep.
  destruct (step s) as [t | k vs | ]; unfold answered_heaps in *;
    [ | | exact I ].
  - destruct Hanswer as (n & t' & Hrun & Ht).
    exists n, t'. split; [ exact Hrun | split; assumption ].
  - destruct Hanswer as (vs' & hp' & rd' & Hexit & Hvs & Hheap).
    exists vs', hp', rd'. split; [ exact Hexit | ].
    split; [ exact (plain_values_lifted _ vs vs' Hstep Hvs) | exact Hheap ].
Qed.

(* The unit lifted refines the unit, the two run from that static
   environment and that heap, if the unit exits with the empty heap
   only: the heap of the unit lifted, which is alike, is then empty
   too. *)
Lemma unit_lifted_refines :
  exits_plain_knowing [] [] e = true ->
  (∀ (k : continuation) (vs : list value) (hp : heap),
    behaves (State e [] [] statics0 hp0 rd0) (Exits k vs hp) ->
    hp = empty_heap) ->
  refines (State e [] [] statics0 hp0 rd0)
    (State (lifted_unit known lets e) [] [] statics0 hp0 rd0).
Proof.
  intros Hplain Hempty.
  apply
    (stuttering_heaps_refines unit_lifted_plain exit_plain
       unit_lifted_plain_step).
  - intros k vs hp last vs' hp' Hexits [Hsame Hheap].
    split; [ exact Hsame | ].
    rewrite (Hempty k vs hp Hexits) in Hheap |- *.
    exact (heap_alike_empty _ hp' Hheap).
  - split;
      [ exact (unit_lifted_initial known lets e statics0 hp0 rd0 [] Hno_code
                 (Forall_nil _))
      | exact (plain_from e statics0 hp0 rd0 Hplain Hno_code) ].
Qed.

End Unit_refines.

Theorem lift_unit_program_refines : lift_unit_program_refines_statement.
Proof.
  intros L e e'. unfold lift_unit. cbv zeta.
  destruct (exits_plain_knowing [] [] e) eqn:Hplain; [ | discriminate ].
  destruct (expr_free_of_heap e) eqn:Hheap_free; [ | discriminate ].
  fold (lifted_unit (lifted_my_closures L) (lifted_lets L) e).
  destruct
    (well_formed_unit (lifting_unit L)
       (lifted_unit (lifted_my_closures L) (lifted_lets L) e))
    eqn:Hwf;
    [ | discriminate ].
  destruct
    (only_closures_of (lifted_my_closures L)
       (lifted_unit (lifted_my_closures L) (lifted_lets L) e))
    eqn:Honly;
    [ | discriminate ].
  intros [= <-]. unfold program_refines.
  exact (unit_lifted_refines (lifted_my_closures L) (lifting_unit L)
           (lifted_lets L) e empty_static_env empty_heap no_reads Hwf Honly
           eq_refl (free_in_empty _)
           (fun id sym _ => statics_ok_empty id sym)
           (fun id sym _ => heap_values_all_empty _) Hplain
           (fun k vs hp =>
              free_of_heap_exits_empty e empty_static_env k vs hp Hheap_free
                eq_refl)).
Qed.

Theorem lift_unit_refines_imports : lift_unit_refines_imports_statement.
Proof.
  intros imports L e e'. unfold lift_unit. cbv zeta.
  destruct (exits_plain_knowing [] [] e) eqn:Hplain; [ | discriminate ].
  destruct (expr_free_of_heap e) eqn:Hheap_free; [ | discriminate ].
  fold (lifted_unit (lifted_my_closures L) (lifted_lets L) e).
  destruct
    (well_formed_unit (lifting_unit L)
       (lifted_unit (lifted_my_closures L) (lifted_lets L) e))
    eqn:Hwf;
    [ | discriminate ].
  destruct
    (only_closures_of (lifted_my_closures L)
       (lifted_unit (lifted_my_closures L) (lifted_lets L) e))
    eqn:Honly;
    [ | discriminate ].
  intros [= <-] Hfree. unfold unit_refines.
  exact (unit_lifted_refines (lifted_my_closures L) (lifting_unit L)
           (lifted_lets L) e (imports_statics imports) empty_heap no_reads Hwf
           Honly eq_refl Hfree
           (fun id sym _ => statics_ok_imports id sym imports)
           (fun id sym _ => heap_values_all_empty _) Hplain
           (fun k vs hp =>
              free_of_heap_exits_empty e (imports_statics imports) k vs hp
                Hheap_free eq_refl)).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)
(* The three laws are proved from the definitions by [equivariance]
   (FlambdaEquivarianceTactic.v), which states and proves, as it goes,
   the laws of the functions that the three are made of. The first
   two need nothing more. The third is of the test with what is
   known, which reads what an expression spoils: a list of
   continuations with positions, renamed as its type says. For it
   three things are given to the tactic:
   - the law of [plain_or] to rewrite with: without it the tactic
     takes cases on that test on each side apart;
   - the law of [forget_conts], used from right to left: under a
     recursive group its arguments are not written as renamed lists;
   - [exn_extra_simples] kept folded, so that its law applies.
   The laws of [params_known] and of [spoils] are stated too, though
   the tactic derives them: stated once, they are not derived again
   in each case that needs them. Each law is stated from the type of
   its function ([equivariance_of]; [Check] shows the statement). *)

(* Used by name in FlambdaLiftAroundProofs.v and
   FlambdaLiftFromCodeProofs.v: the symbol is as it was. *)
Lemma lift_named_rename (r : renaming) (target : lift_to) (d : named) :
  lift_named target (rename_named r d)
  = option_map
      (fun '(bindings, s) => (map (rename_static_binding r) bindings, s))
      (lift_named target d).
Proof. equivariance. Qed.

Theorem lift_expr_equivariant : lift_expr_equivariant_statement.
Proof. equivariance. Qed.

Theorem exits_plain_equivariant : exits_plain_equivariant_statement.
Proof. equivariance. Qed.

#[local] Arguments exn_extra_simples : simpl never.

Lemma plain_or_renamed : equivariance_of plain_or.
Proof. equivariance. Qed.
#[local] Hint Rewrite plain_or_renamed : equivariance.

Lemma forget_conts_renamed : equivariance_of forget_conts.
Proof. equivariance. Qed.

#[local] Ltac eqv_extra ::=
  match goal with
  | |- _ = map _ (forget_conts _ _) =>
    rewrite <- forget_conts_renamed; eqv_same_head; eqv_peel
  end.

Lemma params_known_renamed : equivariance_of params_known.
Proof. equivariance. Qed.
#[local] Hint Rewrite params_known_renamed : equivariance.

Lemma spoils_renamed : equivariance_of spoils.
Proof. equivariance. Qed.
#[local] Hint Rewrite spoils_renamed : equivariance.

Theorem exits_plain_knowing_equivariant :
  exits_plain_knowing_equivariant_statement.
Proof. equivariance. Qed.
