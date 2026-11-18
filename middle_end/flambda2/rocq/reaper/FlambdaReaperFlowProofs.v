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

(* The proofs of FlambdaReaperFlowSpec.v. Nothing here needs review.

   The proof of [flow_sound] has three parts.
   - What the rules carry: a value at a node is at every alias of it
     ([flows_alias]), at a node with any source whose usage implies that
     the first has any usage ([flows_escape]), at the node a field of it
     is read into ([flows_read], and [read_transfer] for any property of
     nodes kept by aliases and escapes; for a local field, with the rule
     of the fix, [flows_read_slot], through
     FlambdaReaperLocalFieldsSpec.local_read_alias_statement), and at the
     parameter of the witness of the code it is given to as an argument
     ([argument_transfer]).
   - What the traversal makes: the facts of a term of the fragment, under
     the nodes of the parameters of the continuations in scope and in
     the code being traversed ([expr_edges]), are in the graph of the
     unit ([traverse_makes], [traverse_unit_facts]). The traversal
     threads an accumulator; a step of it keeps the facts, the symbols,
     the closure deps and the code deps it is given, and adds the code
     deps of the code it binds, so what it makes is in every later
     accumulator ([makes]). The code deps of the unit are those of its
     code, whose ids are distinct, so the code dep that the body of code
     is traversed with is the one that the closure deps read.
   - An invariant of the run ([flow_invariant]): the facts of the
     expression to run are in the graph; the code being run is used, so
     that its conditional facts are aliases; the continuations of the
     state are handlers, each traversed with the continuations of its
     definition in scope, or, in code, the continuations of the call,
     given at the nodes of the code what they are given, a continuation
     with extra arguments at nodes after those of its values, or what is
     left of an over-application, a call of unknown arity yet to be made
     ([conts_ok], [pending]); the values are where [flows] says, with
     what the closure deps give closures ([called_by]); and each partial
     application has fewer groups of arguments than its code has
     parameters ([pa_ok]), which no graph fact can say. Each step keeps
     it, and the cells of the heap hold what [heap_flows] says, and
     partial applications as [pa_ok] says ([heap_ok],
     [step_flow_invariant_heap]); a step other than a [Let] does not
     change the heap ([step_flow_invariant_core]). A [Let] of a primitive
     runs it by [exec_prim] ([exec_prim_cases]): as [eval_prim] does,
     and then as before ([named_flows]); or a primitive of the heap,
     whose fresh location is given the node of its variable
     ([alloc_ok]), or which stores into a cell whose allocation, then,
     has any usage, as what is stored has; or a read of a cell, whose
     allocation [located] gives ([prim_exec_flows]). [located] is
     fixed for the whole run: it is built from the run of [n] steps,
     each location being given the variable of the [Let] that makes it,
     which no later step rebinds, as the heap only grows
     ([located_exists]). A direct call reads the witness of the
     code through the closure, which gives the parameters their
     arguments and the continuations of the call the results
     ([direct_call_ok]). A call of unknown arity has a witness for each
     group of arguments, each read from what the one before returns;
     group by group, they stand for the witnesses of unknown arity of
     the code of the closure ([chain_cover]), which the closure dep gives
     the closure, so the arguments, those that a partial application
     kept included, reach the parameters ([closure_call_ok]); a full
     call returns and raises through the last witness of the code, an
     over-application returns to the rest of the chain, and a partial
     application is at the first result of the last witness of the call
     ([call_ok], [apply_ok]). A call of tupled code gives the tuple to
     the one witness of the code, whose parameter has the fields of the
     tuple read into the parameters of the code ([tupled_call_ok]).
     [called_by] gives the closures a call witness, which is what the
     statement says ([flows_weaken]).
     The invariant is that of module [WithLocal], for any premise
     [slots_read] of the facts of a projection that gives the rule of
     the fix; the names of the run outside it are those before the fix,
     with [slots_read] that no field is local.
   The facts of a term that the proof reads ([expr_edges]) are in every
   graph that has those of the traversal but uses by other nodes than
   variables and symbols and reads of [Is_int] into other nodes than
   variables ([graph_sub], [expr_edges_mono]); so the statement holds of
   any such graph and a solution closed for it, with the rule of the fix
   ([flow_sound_gen_local]), the graph without the two facts that keep a
   tuple alive among them. The traversal of a unit without projection
   reads no value or function slot ([traverse_unit_reads_no_slot]), so
   the premise of the statements before the fix gives the check of the
   fix ([old_premise_local_reads_used]). *)

From Stdlib Require Import ZArith List Bool Arith Lia Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperLocalFields
  FlambdaReaperLocalFieldsSpec.
From OxCamlFlambda2 Require FlambdaHeapProofs FlambdaPrimitiveProofs
  FlambdaTupledProofs FlambdaReaperTupledProofs
  FlambdaReaperLocalFieldsProofs.
Import ListNotations.

(* ======================================================================== *)
(* What the rules carry                                                     *)
(* ======================================================================== *)

Section Carry.

Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Variable kept : value_slot -> bool.
Variable called : node -> code_id -> Prop.
Variable located : location -> option node.
Hypothesis Hc : closed is_local G sol.

Abbreviation as_ n := (s_any_source sol n = true).
Abbreviation au n := (s_any_usage sol n = true).
Abbreviation hs n := (s_has_source sol n = true).
Abbreviation hu n := (s_has_usage sol n = true).
Abbreviation flows_ := (flows G sol kept called located).

(* The case of a location, whose allocation is given by [located]. *)
Ltac location_case :=
  match goal with
  | |- context [located ?l] => destruct (located l); [ | intros [] ]
  end.

(* Leaves the cases of [flows] that are not in the fragment. *)
Ltac not_in_fragment :=
  try (match goal with |- False -> _ => intros [] end).

Lemma au_hu (n : node) : au n -> hu n.
Proof. apply (has_usage_any _ _ _ Hc). Qed.

Lemma as_hs (n : node) : as_ n -> hs n.
Proof. apply (has_source_any _ _ _ Hc). Qed.

(* Along an alias of the solution. *)
Lemma may_come_from_alias (a to_ from : node) :
  s_alias sol to_ from = true ->
  may_come_from sol a from -> may_come_from sol a to_.
Proof.
  intros Hal (Hhs & Hsrc & Has & Hau & Hus).
  split; [ | split; [ | split; [ | split ] ] ].
  - exact (has_source_alias _ _ _ Hc _ _ Hhs Hal).
  - destruct (s_any_source sol to_) eqn:Eto; [ left; reflexivity | right ].
    destruct (s_any_source sol from) eqn:Efrom.
    + rewrite (any_source_alias _ _ _ Hc _ _ Hal Efrom) in Eto.
      discriminate.
    + destruct Hsrc as [Hsrc | Hsrc]; [ discriminate | ].
      exact (sources_alias _ _ _ Hc _ _ _ Hsrc Efrom Hal Eto).
  - intros Ha. exact (any_source_alias _ _ _ Hc _ _ Hal (Has Ha)).
  - intros Hto. apply Hau. exact (any_usage_alias _ _ _ Hc _ _ Hto Hal).
  - intros Hhu u Hu.
    pose proof (has_usage_alias _ _ _ Hc _ _ Hhu Hal) as Hhu'.
    destruct (s_any_usage sol to_) eqn:Eto.
    + right. apply Hau. exact (any_usage_alias _ _ _ Hc _ _ Eto Hal).
    + destruct (s_any_usage sol from) eqn:Efrom.
      * right. apply Hau. reflexivity.
      * apply (Hus Hhu' u).
        exact (usages_alias _ _ _ Hc _ _ _ Hu Eto Hal Efrom).
Qed.

(* To a node with any source that has a usage only if [m] has any
   usage. *)
Lemma may_come_from_escape (a to_ m : node) :
  as_ to_ -> (hu to_ -> au m) ->
  may_come_from sol a m -> may_come_from sol a to_.
Proof.
  intros Has Hm (_ & _ & _ & Hau & _).
  split; [ | split; [ | split; [ | split ] ] ].
  - exact (as_hs _ Has).
  - left. exact Has.
  - intros _. exact Has.
  - intros Hto. apply Hau, Hm, au_hu, Hto.
  - intros Hhu u _. right. apply Hau, Hm, Hhu.
Qed.

(* A value at [from] is at every alias [to_] of it. *)
Lemma flows_alias (v : value) (to_ from : node) :
  s_alias sol to_ from = true -> flows_ v from -> flows_ v to_.
Proof.
  intros Hal. destruct v; simpl; not_in_fragment.
  - intros Has. exact (any_source_alias _ _ _ Hc _ _ Hal Has).
  - intros (sites & Hs & Hrest). exists sites.
    split; [ exact (may_come_from_alias _ _ _ Hal Hs) | exact Hrest ].
  - apply may_come_from_alias. exact Hal.
  - intros (a & Ha & Hrest). exists a.
    split; [ exact (may_come_from_alias _ _ _ Hal Ha) | exact Hrest ].
  - intros (Has & Hall). split.
    + exact (any_source_alias _ _ _ Hc _ _ Hal Has).
    + revert Hall. apply every_impl.
      intros element _ [Hconst | (m & Hm & Hu)]; [ left; exact Hconst | ].
      right. exists m. split; [ exact Hm | ].
      intros Hhu. apply Hu. exact (has_usage_alias _ _ _ Hc _ _ Hhu Hal).
  - intros (a & bn & from' & Ha & Hrest). exists a, bn, from'.
    split; [ exact (may_come_from_alias _ _ _ Hal Ha) | exact Hrest ].
  - location_case. apply may_come_from_alias. exact Hal.
  - intros (m & ws & Hm & Hlen & Hchain & Hex & Hend & Hargs).
    exists m, ws. do 4 (split; [ assumption | ]). split; [ | exact Hargs ].
    intros y Hy. exact (may_come_from_alias _ _ _ Hal (Hend y Hy)).
Qed.

(* A value at [m] is at a node with any source that has a usage only if
   [m] has any usage. *)
Lemma flows_escape (v : value) (to_ m : node) :
  as_ to_ -> (hu to_ -> au m) -> flows_ v m -> flows_ v to_.
Proof.
  intros Has Hm. destruct v; simpl; not_in_fragment.
  - intros _. exact Has.
  - intros (sites & Hs & Hrest). exists sites.
    split; [ exact (may_come_from_escape _ _ _ Has Hm Hs) | exact Hrest ].
  - exact (may_come_from_escape _ _ _ Has Hm).
  - intros (a & Ha & Hrest). exists a.
    split; [ exact (may_come_from_escape _ _ _ Has Hm Ha) | exact Hrest ].
  - intros (_ & Hall). split; [ exact Has | ].
    revert Hall. apply every_impl.
    intros element _ [Hconst | (m' & Hm' & Hu)]; [ left; exact Hconst | ].
    right. exists m'. split; [ exact Hm' | ].
    intros Hhu. apply Hu, au_hu, Hm, Hhu.
  - intros (a & bn & from' & Ha & Hrest). exists a, bn, from'.
    split; [ exact (may_come_from_escape _ _ _ Has Hm Ha) | exact Hrest ].
  - location_case. exact (may_come_from_escape _ _ _ Has Hm).
  - intros (m' & ws & Hm' & Hlen & Hchain & Hex & Hend & Hargs).
    exists m', ws. do 4 (split; [ assumption | ]). split; [ | exact Hargs ].
    intros y Hy. exact (may_come_from_escape _ _ _ Has Hm (Hend y Hy)).
Qed.

(* A node at which a value may be has a source. *)
Lemma flows_hs (v : value) (n : node) : flows_ v n -> hs n.
Proof.
  destruct v; simpl; not_in_fragment.
  - apply as_hs.
  - intros (sites & (Hhs & _) & _). exact Hhs.
  - intros (Hhs & _). exact Hhs.
  - intros (a & (Hhs & _) & _). exact Hhs.
  - intros (Has & _). exact (as_hs _ Has).
  - intros (a & bn & from & (Hhs & _) & _). exact Hhs.
  - location_case. intros (Hhs & _). exact Hhs.
  - intros (m & ws & _ & _ & _ & (y & Hy) & Hend & _).
    exact (proj1 (Hend y Hy)).
Qed.

(* The read of the field [f] of a value that may come from [a]: what
   [a] holds there, at [from], is where it is read into, for any
   property of nodes that aliases and escapes keep. *)
Lemma read_transfer (T : node -> Prop) (a base to_ from : node)
    (f : field) :
  (∀ x y : node, s_alias sol x y = true -> T y -> T x) ->
  (∀ x m : node, as_ x -> (hu x -> au m) -> T m -> T x) ->
  is_local f = false ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  T from -> T to_.
Proof.
  intros Talias Tescape Hlocal Hcons Hacc (_ & Hsrc & _ & Hau & Hus) Hv.
  destruct (s_any_source sol base) eqn:Eb.
  - pose proof (any_source_accessor _ _ _ Hc _ _ _ Eb Hacc Hlocal) as Has.
    destruct (s_any_usage sol a) eqn:Ea.
    + apply (Tescape to_ from Has); [ | exact Hv ].
      intros _. exact (any_usage_constructor _ _ _ Hc _ _ _ Ea Hcons Hlocal).
    + destruct (s_has_usage sol to_) eqn:Ehu.
      * pose proof (has_usage_accessor _ _ _ Hc _ _ _ Ehu Hacc) as Hhu.
        assert (Eau : s_any_usage sol base = false).
        { destruct (s_any_usage sol base);
          [ discriminate (Hau eq_refl) | reflexivity ]. }
        pose proof (usages_accessor _ _ _ Hc _ _ _ Hacc Eau) as Hbb.
        destruct (Hus Hhu base Hbb) as [Hab | Hab]; [ | congruence ].
        apply (Talias to_ from); [ | exact Hv ].
        exact (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hab Ea Hacc).
      * apply (Tescape to_ from Has); [ | exact Hv ].
        intros Hhu. congruence.
  - destruct Hsrc as [Hsrc | Hsrc]; [ discriminate | ].
    apply (Talias to_ from); [ | exact Hv ].
    exact (alias_accessor_source _ _ _ Hc _ _ _ _ _ Hacc Hsrc Eb Hcons).
Qed.

Lemma flows_read (a base to_ from : node) (f : field) (v : value) :
  is_local f = false ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  flows_ v from -> flows_ v to_.
Proof.
  apply (read_transfer (flows_ v)); [ apply flows_alias | apply flows_escape ].
Qed.

Lemma may_come_from_read (c a base to_ from : node) (f : field) :
  is_local f = false ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  may_come_from sol c from -> may_come_from sol c to_.
Proof.
  apply (read_transfer (may_come_from sol c));
    [ intros x y; apply may_come_from_alias
    | intros x m; apply may_come_from_escape ].
Qed.

(* A read into a node with any usage gives what is read any usage. *)
Lemma read_any_usage (a base to_ from : node) (f : field) :
  is_local f = false ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  au to_ -> au from.
Proof.
  intros Hlocal Hcons Hacc (_ & Hsrc & _ & Hau & Hus) Hto.
  destruct (s_any_source sol base) eqn:Eb.
  - destruct (s_any_usage sol a) eqn:Ea.
    + exact (any_usage_constructor _ _ _ Hc _ _ _ Ea Hcons Hlocal).
    + pose proof (has_usage_accessor _ _ _ Hc _ _ _ (au_hu _ Hto) Hacc)
        as Hhu.
      assert (Eau : s_any_usage sol base = false).
      { destruct (s_any_usage sol base);
        [ discriminate (Hau eq_refl) | reflexivity ]. }
      pose proof (usages_accessor _ _ _ Hc _ _ _ Hacc Eau) as Hbb.
      destruct (Hus Hhu base Hbb) as [Hab | Hab]; [ | congruence ].
      exact (any_usage_alias _ _ _ Hc _ _ Hto
               (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hab Ea Hacc)).
  - destruct Hsrc as [Hsrc | Hsrc]; [ discriminate | ].
    exact (any_usage_alias _ _ _ Hc _ _ Hto
             (alias_accessor_source _ _ _ Hc _ _ _ _ _ Hacc Hsrc Eb Hcons)).
Qed.

(* The argument [from] of a call whose witness [base] may come from the
   witness [w] of code is where the parameter [to_] of that witness is,
   for any property of nodes that aliases and escapes keep. *)
Lemma argument_transfer (T : node -> Prop) (w base to_ from : node)
    (i : nat) :
  (∀ x y : node, s_alias sol x y = true -> T y -> T x) ->
  (∀ x m : node, as_ x -> (hu x -> au m) -> T m -> T x) ->
  may_come_from sol w base ->
  g_argument G from i base ->
  g_parameter G w i to_ ->
  hs from ->
  T from -> T to_.
Proof.
  intros Talias Tescape (_ & Hsrc & _ & Hau & Hus) Harg Hpar Hhs Hv.
  destruct (s_any_source sol base) eqn:Eb.
  - destruct (s_any_usage sol w) eqn:Ew.
    + apply (Tescape to_ from); [ | | exact Hv ].
      * exact (any_source_parameter _ _ _ Hc _ _ _ Ew Hpar).
      * intros _. exact (any_usage_argument _ _ _ Hc _ _ _ Eb Harg).
    + pose proof (has_usage_argument _ _ _ Hc _ _ _ Hhs Harg) as Hhu.
      assert (Eau : s_any_usage sol base = false).
      { destruct (s_any_usage sol base);
        [ discriminate (Hau eq_refl) | reflexivity ]. }
      pose proof (usages_argument _ _ _ Hc _ _ _ Harg Eau) as Hbb.
      destruct (Hus Hhu base Hbb) as [Hwb | Hwb]; [ | congruence ].
      apply (Talias to_ from); [ | exact Hv ].
      exact (alias_parameter_usage _ _ _ Hc _ _ _ _ _ Hpar Hwb Ew Harg).
  - destruct Hsrc as [Hsrc | Hsrc]; [ discriminate | ].
    apply (Talias to_ from); [ | exact Hv ].
    exact (alias_argument_source _ _ _ Hc _ _ _ _ _ Harg Hsrc Eb Hpar).
Qed.

(* A read into a node that has a usage uses the field of every
   allocation that the value read may come from. *)
Lemma field_used_read (a base to_ from : node) (f : field) :
  is_local f = false ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  hu to_ -> s_field_used sol a f = true.
Proof.
  intros Hlocal Hcons Hacc (_ & _ & _ & Hau & Hus) Hhu.
  destruct (s_any_usage sol a) eqn:Ea.
  - exact (field_used_any_usage _ _ _ Hc _ _ _ Hcons Ea Hlocal).
  - pose proof (has_usage_accessor _ _ _ Hc _ _ _ Hhu Hacc) as Hhub.
    assert (Eau : s_any_usage sol base = false).
    { destruct (s_any_usage sol base);
      [ discriminate (Hau eq_refl) | reflexivity ]. }
    pose proof (usages_accessor _ _ _ Hc _ _ _ Hacc Eau) as Hbb.
    destruct (Hus Hhub base Hbb) as [Hab | Hab]; [ | congruence ].
    exact (field_used_usage _ _ _ Hc _ _ _ _ _ Hcons Hab Hacc Hhu).
Qed.

(* The same of a read of any field, with the rule of the fix
   ([local_reads_used]) if the field is local
   (FlambdaReaperLocalFieldsSpec.local_read_alias_statement). *)
Lemma flows_read_slot (a base to_ from : node) (f : field) (v : value) :
  local_reads_used is_local G sol ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  flows_ v from -> flows_ v to_.
Proof.
  intros Hlrs Hcons Hacc Hmay Hv. destruct (is_local f) eqn:Hl.
  - apply (flows_alias v to_ from); [ | exact Hv ].
    exact (FlambdaReaperLocalFieldsProofs.local_read_alias is_local G sol a
             base to_ from f (conj Hc Hlrs) Hl Hcons Hacc Hmay).
  - exact (flows_read a base to_ from f v Hl Hcons Hacc Hmay Hv).
Qed.

(* A read into a node that has a usage uses the field, local or not; this
   needs no fix
   (FlambdaReaperLocalFieldsSpec.local_read_field_used_statement). *)
Lemma field_used_read_slot (a base to_ from : node) (f : field) :
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  hu to_ -> s_field_used sol a f = true.
Proof.
  intros Hcons Hacc Hmay Hhu. destruct (is_local f) eqn:Hl.
  - exact (FlambdaReaperLocalFieldsProofs.local_read_field_used is_local G
             sol a base to_ from f Hc Hl Hcons Hacc Hmay Hhu).
  - exact (field_used_read a base to_ from f Hl Hcons Hacc Hmay Hhu).
Qed.

End Carry.

Theorem flows_has_source : flows_has_source_statement.
Proof.
  intros is_local G sol kept called located v n Hc.
  exact (flows_hs is_local G sol kept called located Hc v n).
Qed.

Theorem read_field_used : read_field_used_statement.
Proof.
  intros is_local G sol a base to_ from f Hc.
  exact (field_used_read is_local G sol Hc a base to_ from f).
Qed.

Theorem read_flows : read_flows_statement.
Proof.
  intros is_local G sol kept called located a base to_ from f v Hc.
  exact (flows_read is_local G sol kept called located Hc a base to_ from f
           v).
Qed.

(* ======================================================================== *)
(* The facts of a term                                                      *)
(* ======================================================================== *)

(* What the traversal puts into the graph for a term of the fragment, in
   the code [ctx] (none at toplevel), under [ce], the nodes of the
   parameters of the continuations in scope. [noted] are the symbols that
   went through [simple_to_node], [deps] the closure deps, [cds] the code
   deps, [keep] the value slots kept, [codes] and [symbols] the code and
   the symbols that the unit binds. Only the facts that the proofs read
   are here. *)

(* The groups of the arguments of a call, as the traversal cuts them. *)
Definition call_groups (ap : apply_expr) : list (list simple) :=
  at_least_one_group (group_by_parameter (ap_args_arity ap) (ap_args ap)).

Section Edges.

Variable G : graph.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable keep : value_slot -> bool.
Variable codes : list (code_id * code).
Variable symbols : list symbol.
(* Whether the reads of value and function slots are not local. *)
Variable slots_read : Prop.
(* Whether the primitives of the heap are in the term. *)
Variable heap_prims : Prop.

Definition simple_noted (s : simple) : Prop :=
  match s with
  | Symbol sym => noted sym
  | Var _ | Const _ => True
  end.

Definition simples_noted (l : list simple) : Prop :=
  ∀ s : simple, s ∈ l -> simple_noted s.

Definition block_edges (base : node) (fields : list simple) : Prop :=
  (∀ (i : nat) (s : simple),
     nth_error fields i = Some s ->
     g_constructor G base (Field_block i Value) (node_of_simple s))
  ∧ g_constructor G base Field_is_int Node_all_constants
  ∧ g_constructor G base Field_get_tag Node_all_constants
  ∧ simples_noted fields.

(* The facts of a static boxed number of [s] ([boxed_number_facts]). *)
Definition boxed_edges (s : symbol) (contents : simple) : Prop :=
  match boxed_number_kind contents with
  | Some bn =>
    g_constructor G (Node_symbol s) (Field_boxed_number bn) Node_all_constants
  | None =>
    g_alias G (Node_symbol s) Node_all_constants
    ∧ ∀ x : variable,
        contents = Var x -> g_use G (Node_symbol s) (Node_variable x)
  end.

Definition accessor_edges (x : variable) (f : field) (s : simple) : Prop :=
  g_accessor G (Node_variable x) f (node_of_simple s) ∧ simple_noted s.

(* The primitives that [traverse_prim] reads in a way of their own. *)
Definition prim_classified (p : prim) (args : list simple) : bool :=
  match p, args with
  | Variadic (Make_block _), _ | Heap (Make_block_with_identity _ _), _ =>
    true
  | Unary (Block_load _ _ _), [_] | Unary (Is_int true), [_]
  | Unary Get_tag, [_] | Unary (Box_number _), [_]
  | Unary (Unbox_number _), [_] => true
  | Region_primitive Local_allocation, _ => true
  | _, _ => false
  end.

Definition classified_prim_edges (x : variable) (p : prim)
    (args : list simple) : Prop :=
  match p, args with
  | Variadic (Make_block _), fields
  | Heap (Make_block_with_identity _ _), fields =>
    block_edges (Node_variable x) fields
  | Unary (Block_load _ _ field), [block] =>
    accessor_edges x (Field_block (Z.to_nat field) Value) block
  | Unary (Is_int true), [arg] => accessor_edges x Field_is_int arg
  | Unary Get_tag, [arg] => accessor_edges x Field_get_tag arg
  | Unary (Box_number bn), [contents] =>
    g_constructor G (Node_variable x) (Field_boxed_number bn)
      (node_of_simple contents)
    ∧ simple_noted contents
  | Unary (Unbox_number bn), [arg] =>
    accessor_edges x (Field_boxed_number bn) arg
  | _, _ => True
  end.

Definition default_prim_edges (x : variable) (args : list simple) : Prop :=
  g_any_source G (Node_variable x)
  ∧ ∀ from : node,
      from ∈ free_names_of_simples args -> g_use G (Node_variable x) from.

(* [cond_any_usage] in the code [ctx]. *)
Definition cond_use_edge (ctx : option code_id) (n : node) : Prop :=
  match ctx with
  | None => g_any_usage G n
  | Some id => g_use G (Node_code_id id) n
  end.

(* A load of a field that may change: its variable has any source. *)
Definition prim_loads_mutable (p : prim) : bool :=
  match p with
  | Unary (Block_load _ Mutable _) => true
  | _ => false
  end.

(* Of a primitive, in the code [ctx]: not of the heap unless [heap]
   ([named_in_flow_fragment]); used by the code where it has arbitrary
   effects; with any source if it loads a mutable field; and its facts,
   by the case of [traverse_prim]. *)
Definition prim_edges (ctx : option code_id) (x : variable) (p : prim)
    (args : list simple) : Prop :=
  ((prim_of_heap p = false ∨ heap_prims)
   ∧ (prim_has_arbitrary_effects p = true ->
      cond_use_edge ctx (Node_variable x))
   ∧ (prim_loads_mutable p = true -> g_any_source G (Node_variable x)))
  ∧ if prim_classified p args
    then classified_prim_edges x p args
    else default_prim_edges x args.

(* The closure of a set at the node [b], of the code [id]: its closure
   dep, its function slots and the value slots kept. *)
Definition closure_edges (b : node) (id : code_id)
    (function_decls : list (function_slot * code_id))
    (sites : function_slot -> node)
    (value_slots : list (value_slot * simple)) : Prop :=
  deps {| cld_let_bound_name_of_the_closure := b; cld_closure_code_id := id |}
  ∧ (∀ (g : function_slot) (id' : code_id),
       lookup g function_decls = Some id' ->
       g_constructor G b (Field_function_slot g) (sites g))
  ∧ (∀ (w : value_slot) (s : simple),
       lookup w value_slots = Some s -> keep w = true ->
       g_constructor G b (Field_value_slot w) (node_of_simple s)
       ∧ simple_noted s).

Definition set_edges (x : variable) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) : Prop :=
  ∀ id0 : code_id,
    lookup slot function_decls = Some id0 ->
    ∃ sites : function_slot -> node,
      sites slot = Node_variable x
      ∧ ∀ (f : function_slot) (id : code_id),
          lookup f function_decls = Some id ->
          closure_edges (sites f) id function_decls sites value_slots.

Definition named_edges (ctx : option code_id) (x : variable)
    (defining_expr : named) : Prop :=
  match defining_expr with
  | Simple s =>
    g_alias G (Node_variable x) (node_of_simple s) ∧ simple_noted s
  | Prim p args => prim_edges ctx x p args
  | Closure_of_set slot function_decls value_slots =>
    set_edges x slot function_decls value_slots
  | Project_value_slot_from _ w closure =>
    accessor_edges x (Field_value_slot w) closure ∧ keep w = true
    ∧ slots_read
  | Project_function_slot _ move_to closure =>
    accessor_edges x (Field_function_slot move_to) closure ∧ slots_read
  | Set_of_closures _ _ | Project_value_slot _ _ => False
  end.

(* A static closure of a set, bound to [s]. *)
Definition static_set_edges (s : symbol) (slot : function_slot)
    (closure_symbols : list (function_slot * symbol))
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) : Prop :=
  is_bound slot function_decls = true
  ∧ (∀ id : code_id,
     lookup slot function_decls = Some id ->
     deps {| cld_let_bound_name_of_the_closure := Node_symbol s;
             cld_closure_code_id := id |})
  ∧ (∀ (g : function_slot) (s' : symbol),
       lookup g closure_symbols = Some s' ->
       g_constructor G (Node_symbol s) (Field_function_slot g)
         (Node_symbol s')
       ∧ among s' symbols = true)
  ∧ (∀ (w : value_slot) (x : simple),
       lookup w value_slots = Some x -> keep w = true ->
       g_constructor G (Node_symbol s) (Field_value_slot w) (node_of_simple x)
       ∧ simple_noted x).

(* The witness of a full call of the code [id], prepared with [cd]. *)
Definition known_witness_edges (id : code_id) (cd : code_dep) : Prop :=
  (∀ (i : nat) (p : node),
     nth_error (cd_params cd) i = Some p ->
     g_parameter G (cd_known_arity_call_witness cd) i p)
  ∧ (∀ (i : nat) (r : node),
       nth_error (cd_return cd) i = Some r ->
       g_constructor G (cd_known_arity_call_witness cd)
         (Field_return_of_call (Normal i)) r)
  ∧ g_constructor G (cd_known_arity_call_witness cd)
      (Field_return_of_call Exn) (cd_exn cd)
  ∧ g_constructor G (cd_known_arity_call_witness cd)
      Field_code_id_of_call_witness (Node_code_id id)
  ∧ g_any_source G (Node_code_id id).

Definition apply_cont_edges (ce : finite_map (list node)) (ac : apply_cont)
    : Prop :=
  (∀ params : list node,
     lookup (ac_continuation ac) ce = Some params ->
     ∀ (p : node) (s : simple),
       (p, s) ∈ combine params (ac_args ac) ->
       g_alias G p (node_of_simple s))
  ∧ simples_noted (ac_args ac).

(* [cond_alias] in the code [ctx]. *)
Definition cond_alias_edge (ctx : option code_id) (to_ from : node) : Prop :=
  match ctx with
  | None => g_alias G to_ from
  | Some id => g_propagate G (Node_code_id id) to_ from
  end.

(* The witnesses of a call of unknown arity, one for each group of
   arguments ([unknown_arity_apply_deps]): each takes its group, and
   gives the exception; the last gives the results, and each other a
   first result whose call witness of unknown arity is the next one. *)
Fixpoint caller_chain_edges (returns : list node) (exn called : node)
    (l : list (list simple * node)) : Prop :=
  match l with
  | [] => True
  | (first, w) :: rest =>
    (∀ (k : nat) (s : simple),
       nth_error first k = Some s -> g_argument G (node_of_simple s) k w)
    ∧ g_accessor G exn (Field_return_of_call Exn) w
    ∧ g_accessor G called Field_code_id_of_call_witness w
    ∧ match rest with
      | [] =>
        ∀ (i : nat) (r : node),
          nth_error returns i = Some r ->
          g_accessor G r (Field_return_of_call (Normal i)) w
      | (_, w') :: _ =>
        ∃ v : node,
          g_accessor G v (Field_return_of_call (Normal 0)) w
          ∧ g_accessor G w' (Field_call_witness Unknown_arity_code_pointer) v
      end
    ∧ caller_chain_edges returns exn called rest
  end.

(* The witnesses of unknown arity of the code [id], one for each group
   of parameters ([unknown_arity_witness_deps]): the last gives the
   results and the exception, each other a first result whose call
   witness of unknown arity is the next one. *)
Fixpoint code_chain_edges (id : code_id) (returns : list node) (exn : node)
    (l : list (list node * node)) : Prop :=
  match l with
  | [] => True
  | (first, w) :: rest =>
    (∀ (k : nat) (p : node), nth_error first k = Some p -> g_parameter G w k p)
    ∧ g_constructor G w Field_code_id_of_call_witness (Node_code_id id)
    ∧ match rest with
      | [] =>
        g_constructor G w (Field_return_of_call Exn) exn
        ∧ ∀ (i : nat) (r : node),
            nth_error returns i = Some r ->
            g_constructor G w (Field_return_of_call (Normal i)) r
      | (_, w') :: _ =>
        ∃ v : node,
          g_constructor G w (Field_return_of_call (Normal 0)) v
          ∧ g_constructor G v (Field_call_witness Unknown_arity_code_pointer)
              w'
      end
    ∧ code_chain_edges id returns exn rest
  end.

(* The witness of unknown arity of the tupled code [id]
   ([create_unknown_arity_tupled_call_witnesses]): it gives the results
   and the exception, and the parameters of the code are read off the
   fields of its one parameter. The two facts that keep the tuple alive
   are not here: nothing below reads them. *)
Definition tupled_witness_edges (id : code_id) (cd : code_dep) : Prop :=
  ∃ w untuple : node,
    cd_unknown_arity_call_witnesses cd = [w]
    ∧ (∀ (i : nat) (r : node),
         nth_error (cd_return cd) i = Some r ->
         g_constructor G w (Field_return_of_call (Normal i)) r)
    ∧ g_constructor G w (Field_return_of_call Exn) (cd_exn cd)
    ∧ g_constructor G w Field_code_id_of_call_witness (Node_code_id id)
    ∧ g_parameter G w 0 untuple
    ∧ ∀ (i : nat) (p : node),
        nth_error (cd_params cd) i = Some p ->
        g_accessor G p (Field_block i Value) untuple.

(* A call, with the nodes of its results ([returns]) and of its
   exception ([exn]): a direct call, or one of known arity, reads them
   off its witness, which reads the call witness of known arity of the
   callee; one of unknown arity off its chain of witnesses, the first of
   which reads the call witness of unknown arity of the callee. *)
Definition call_edges (ctx : option code_id) (ap : apply_expr)
    (returns : list node) (exn : node) : Prop :=
  match ap_call_kind ap with
  | Direct _ | Indirect_known_arity _ =>
    ∃ witness apply_node called_node : node,
      (∀ (i : nat) (s : simple),
         nth_error (ap_args ap) i = Some s ->
         g_argument G (node_of_simple s) i witness)
      ∧ (∀ (i : nat) (r : node),
           nth_error returns i = Some r ->
           g_accessor G r (Field_return_of_call (Normal i)) witness)
      ∧ g_accessor G exn (Field_return_of_call Exn) witness
      ∧ g_accessor G called_node Field_code_id_of_call_witness witness
      ∧ g_any_usage G called_node
      ∧ cond_alias_edge ctx witness apply_node
      ∧ g_accessor G apply_node
          (Field_call_witness Known_arity_code_pointer)
          (node_of_simple (ap_callee ap))
  | Indirect_unknown_arity =>
    ∃ (ws : list node) (apply_node called_node : node),
      length ws = length (call_groups ap)
      ∧ caller_chain_edges returns exn called_node (combine (call_groups ap) ws)
      ∧ g_any_usage G called_node
      ∧ (∀ w : node, hd_error ws = Some w -> cond_alias_edge ctx w apply_node)
      ∧ g_accessor G apply_node
          (Field_call_witness Unknown_arity_code_pointer)
          (node_of_simple (ap_callee ap))
  end.

(* A call: its continuations, as found under [ce], the first parameter
   of the exception continuation being the exception and the others
   aliased to the extra arguments, and what the call reads. *)
Definition apply_edges (ctx : option code_id) (ce : finite_map (list node))
    (ap : apply_expr) : Prop :=
  apply_in_flow_fragment codes ap = true
  ∧ simple_noted (ap_callee ap)
  ∧ simples_noted (ap_args ap)
  ∧ ∃ (returns : list node) (exn : node),
      (∀ params : list node,
         lookup (ap_continuation ap) ce = Some params -> returns = params)
      ∧ (∀ (p : node) (params : list node),
           lookup (ap_exn_continuation ap) ce = Some (p :: params) ->
           exn = p
           ∧ simples_noted (exn_extra_simples ap)
           ∧ ∀ (q : node) (s : simple),
               (q, s) ∈ combine params (exn_extra_simples ap) ->
               g_alias G q (node_of_simple s))
      ∧ call_edges ctx ap returns exn.

(* The continuations of a recursive group in scope. *)
Definition rec_conts (ce : finite_map (list node))
    (handlers : list (continuation * continuation_handler))
    : finite_map (list node) :=
  map (fun '(k, h) => (k, handler_params h)) handlers ++ ce.

(* The nodes of the continuations of the body of code. *)
Definition code_conts (c : code) (cd : code_dep) : finite_map (list node) :=
  [(code_exn_continuation c, [cd_exn cd]);
   (code_return_continuation c, cd_return cd)].

Fixpoint expr_edges (ctx : option code_id) (ce : finite_map (list node))
    (e : expr) : Prop :=
  match e with
  | Let x _ defining_expr body =>
    named_edges ctx x defining_expr ∧ expr_edges ctx ce body
  | Let_static bindings body =>
    every binding_edges bindings ∧ expr_edges ctx ce body
  | Let_cont_non_recursive k h body =>
    handler_edges ctx ce h
    ∧ expr_edges ctx ((k, handler_params h) :: ce) body
  | Let_cont_recursive handlers body =>
    every
      (fun '(_, h) => handler_edges ctx (rec_conts ce handlers) h) handlers
    ∧ expr_edges ctx (rec_conts ce handlers) body
  | Apply ap => apply_edges ctx ce ap
  | Apply_cont ac => apply_cont_edges ce ac
  | Switch _ arms =>
    every (fun '(_, ac) => apply_cont_edges ce ac) (Keyed.compact Z.eqb arms)
  | Invalid => True
  end
with handler_edges (ctx : option code_id) (ce : finite_map (list node))
    (h : continuation_handler) : Prop :=
  match h with
  | Handler _ _ handler => expr_edges ctx ce handler
  end
with binding_edges (binding : static_binding) : Prop :=
  match binding with
  | Bind_symbol s (Static_const_block _ fields) =>
    block_edges (Node_symbol s) fields
  | Bind_symbol s (Static_const_immutable_value_array []) =>
    g_alias G (Node_symbol s) Node_all_constants
  | Bind_symbol s (Static_const_immutable_value_array elements) =>
    block_edges (Node_symbol s) elements
  | Bind_symbol s (Static_const_boxed_number contents) =>
    boxed_edges s contents
  | Bind_symbol s (Static_const_immutable_string _) =>
    g_alias G (Node_symbol s) Node_all_constants
  | Bind_symbol s
      (Static_const_closure_of_set slot closure_symbols function_decls
         value_slots) =>
    static_set_edges s slot closure_symbols function_decls value_slots
  | Bind_symbol _ _ => False
  | Bind_code id c => (id, c) ∈ codes ∧ code_edges id c
  | Bind_deleted_code _ => True
  end
with code_edges (id : code_id) (c : code) : Prop :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    ∃ cd : code_dep,
      (id, cd) ∈ cds
      ∧ known_witness_edges id cd
      ∧ (cm_is_tupled metadata = false ->
         length (cd_unknown_arity_call_witnesses cd)
         = length (at_least_one_group
                     (group_by_parameter (cm_params_arity metadata)
                        (cd_params cd)))
         ∧ code_chain_edges id (cd_return cd) (cd_exn cd)
             (combine
                (at_least_one_group
                   (group_by_parameter (cm_params_arity metadata)
                      (cd_params cd)))
                (cd_unknown_arity_call_witnesses cd)))
      ∧ (cm_is_tupled metadata = true -> tupled_witness_edges id cd)
      ∧ length (cd_params cd) = length (unarize (cm_params_arity metadata))
      ∧ length (cd_return cd) = length (unarize (cm_result_arity metadata))
      ∧ g_alias G (Node_variable my_closure) (cd_my_closure cd)
      ∧ (∀ (p : variable) (q : node),
           (p, q) ∈ combine params (cd_params cd) ->
           g_alias G (Node_variable p) q)
      ∧ return_continuation <> exn_continuation
      ∧ binds_statics body = false
      ∧ conts_counted
          [(exn_continuation, 1);
           (return_continuation, length (unarize (cm_result_arity metadata)))]
          body
        = true
      ∧ expr_edges (Some id)
          [(exn_continuation, [cd_exn cd]);
           (return_continuation, cd_return cd)]
          body
  end.

End Edges.

(* Induction on the terms of the fragment, with the handlers of a
   recursive group and the bodies of the code that a [Let_static]
   binds. *)
Lemma expr_handler_ind (P : expr -> Prop) (Q : continuation_handler -> Prop) :
  (∀ x kind defining_expr body, P body -> P (Let x kind defining_expr body)) ->
  (∀ bindings body,
     (∀ (id : code_id) (c : code), Bind_code id c ∈ bindings ->
        P (code_body c)) ->
     P body -> P (Let_static bindings body)) ->
  (∀ k h body, Q h -> P body -> P (Let_cont_non_recursive k h body)) ->
  (∀ handlers body,
     (∀ k h, (k, h) ∈ handlers -> Q h) -> P body ->
     P (Let_cont_recursive handlers body)) ->
  (∀ ap, P (Apply ap)) ->
  (∀ ac, P (Apply_cont ac)) ->
  (∀ scrutinee arms, P (Switch scrutinee arms)) ->
  P Invalid ->
  (∀ params metadata handler,
     P handler -> Q (Handler params metadata handler)) ->
  (∀ e, P e) ∧ (∀ h, Q h).
Proof.
  intros Hlet Hstatic Hnonrec Hrec Happly Happly_cont Hswitch Hinvalid
    Hhandler.
  assert (He : ∀ e, P e).
  { fix IH 1 with (IHh (h : continuation_handler) {struct h} : Q h).
    - intros [x kind d body | bindings body | k h body | handlers body | ap
             | ac | scrutinee arms | ].
      + apply Hlet, IH.
      + apply Hstatic; [ | apply IH ].
        exact
          ((fix go (l : list static_binding)
              : ∀ id c, Bind_code id c ∈ l -> P (code_body c) :=
              match l with
              | [] => fun id c Hin => match Hin with end
              | b :: l' =>
                fun id c Hin =>
                  match Hin with
                  | or_introl Heq =>
                    eq_ind b
                      (fun b0 =>
                         match b0 with
                         | Bind_code _ c0 => P (code_body c0)
                         | _ => True
                         end)
                      (match b as b0
                         return
                           (match b0 with
                            | Bind_code _ c0 => P (code_body c0)
                            | _ => True
                            end)
                       with
                       | Bind_code _ c0 => IH (code_body c0)
                       | _ => I
                       end)
                      (Bind_code id c) Heq
                  | or_intror Hin' => go l' id c Hin'
                  end
              end) bindings).
      + apply Hnonrec; [ apply IHh | apply IH ].
      + apply Hrec; [ | apply IH ].
        exact
          ((fix go (l : list (continuation * continuation_handler))
              : ∀ k h, (k, h) ∈ l -> Q h :=
              match l with
              | [] => fun k h Hin => match Hin with end
              | (k', h') :: l' =>
                fun k h Hin =>
                  match Hin with
                  | or_introl Heq =>
                    eq_ind (k', h') (fun p => Q (snd p)) (IHh h') (k, h) Heq
                  | or_intror Hin' => go l' k h Hin'
                  end
              end) handlers).
      + apply Happly.
      + apply Happly_cont.
      + apply Hswitch.
      + exact Hinvalid.
    - intros [params metadata handler]. apply Hhandler, IH. }
  split; [ exact He | ].
  intros [params metadata handler]. apply Hhandler, He.
Qed.

(* ======================================================================== *)
(* The accumulator                                                          *)
(* ======================================================================== *)

(* An accumulator that has the facts, the symbols, the closure deps and
   the code deps of another. *)
Definition acc_le (a a' : acc) : Prop :=
  (∀ f : fact, f ∈ acc_facts a -> f ∈ acc_facts a')
  ∧ (∀ s : symbol, s ∈ acc_symbols a -> s ∈ acc_symbols a')
  ∧ (∀ d : closure_dep, d ∈ acc_closure_deps a -> d ∈ acc_closure_deps a')
  ∧ (∀ x : code_id * code_dep, x ∈ acc_code_deps a -> x ∈ acc_code_deps a').

Lemma acc_le_refl (a : acc) : acc_le a a.
Proof. repeat split; auto. Qed.

Lemma acc_le_trans (a a' a'' : acc) :
  acc_le a a' -> acc_le a' a'' -> acc_le a a''.
Proof.
  intros (H1 & H2 & H3 & H4) (H5 & H6 & H7 & H8). repeat split; auto.
Qed.

(* A step that adds no code dep. *)
Definition steady (a a' : acc) : Prop :=
  acc_le a a' ∧ acc_code_deps a' = acc_code_deps a.

Lemma steady_refl (a : acc) : steady a a.
Proof. split; [ apply acc_le_refl | reflexivity ]. Qed.

Lemma steady_trans (a a' a'' : acc) :
  steady a a' -> steady a' a'' -> steady a a''.
Proof.
  intros [H1 H2] [H3 H4]. split; [ exact (acc_le_trans _ _ _ H1 H3) |
    congruence ].
Qed.

(* A step that adds the code deps of [ids], in some order. *)
Definition adds (ids : list code_id) (a a' : acc) : Prop :=
  acc_le a a'
  ∧ ∃ l : finite_map code_dep,
      acc_code_deps a' = l ++ acc_code_deps a
      ∧ Permutation (map fst l) ids.

Lemma steady_adds (a a' : acc) : steady a a' -> adds [] a a'.
Proof. intros [Hle Hcds]. split; [ exact Hle | exists []; auto ]. Qed.

Lemma adds_trans (ids ids' : list code_id) (a a' a'' : acc) :
  adds ids a a' -> adds ids' a' a'' -> adds (ids ++ ids') a a''.
Proof.
  intros [Hle (l & Hl & Hperm)] [Hle' (l' & Hl' & Hperm')].
  split; [ exact (acc_le_trans _ _ _ Hle Hle') | ].
  exists (l' ++ l). split; [ rewrite Hl', Hl, app_assoc; reflexivity | ].
  rewrite map_app. eapply Permutation_trans; [ apply Permutation_app_comm | ].
  apply Permutation_app; assumption.
Qed.

(* The graph of the facts of an accumulator, and its symbols. *)
Definition acc_graph (a : acc) : graph :=
  graph_of_facts (facts_of_list (acc_facts a)).

Definition acc_noted (a : acc) (s : symbol) : Prop := s ∈ acc_symbols a.

Definition acc_deps (a : acc) (d : closure_dep) : Prop :=
  d ∈ acc_closure_deps a.

(* A step of the traversal that adds the code deps of [ids] and makes
   what [P] asks of any later accumulator. *)
Definition makes (ids : list code_id) (step : acc -> acc) (P : acc -> Prop)
    : Prop :=
  ∀ a : acc, adds ids a (step a) ∧ ∀ a', acc_le (step a) a' -> P a'.

Lemma makes_compose (ids ids' : list code_id) (step step' : acc -> acc)
    (P P' : acc -> Prop) :
  makes ids step P -> makes ids' step' P' ->
  makes (ids ++ ids') (fun a => step' (step a)) (fun a' => P a' ∧ P' a').
Proof.
  intros H H' a. destruct (H a) as [Hadds HP].
  destruct (H' (step a)) as [Hadds' HP']. split.
  - exact (adds_trans _ _ _ _ _ Hadds Hadds').
  - intros a' Hle''. split.
    + apply HP. exact (acc_le_trans _ _ _ (proj1 Hadds') Hle'').
    + apply HP'. exact Hle''.
Qed.

Lemma makes_weaken (ids : list code_id) (step : acc -> acc)
    (P P' : acc -> Prop) :
  (∀ a', P a' -> P' a') -> makes ids step P -> makes ids step P'.
Proof.
  intros Himpl H a. destruct (H a) as [Hadds HP].
  split; [ exact Hadds | intros a' Hle'; apply Himpl, HP, Hle' ].
Qed.

Lemma makes_ext (ids : list code_id) (step step' : acc -> acc)
    (P : acc -> Prop) :
  (∀ a, step a = step' a) -> makes ids step P -> makes ids step' P.
Proof. intros Heq H a. rewrite <- Heq. apply H. Qed.

Lemma makes_perm (ids ids' : list code_id) (step : acc -> acc)
    (P : acc -> Prop) :
  Permutation ids ids' -> makes ids step P -> makes ids' step P.
Proof.
  intros Hperm H a. destruct (H a) as [[Hle (l & Hl & Hp)] HP].
  split; [ split; [ exact Hle | exists l; split;
    [ exact Hl | exact (Permutation_trans Hp Hperm) ] ] | exact HP ].
Qed.

(* A step that adds nothing but facts, symbols and closure deps. *)
Lemma makes_steady (step : acc -> acc) (P : acc -> Prop) :
  (∀ a, steady a (step a)) ->
  (∀ a a', acc_le (step a) a' -> P a') ->
  makes [] step P.
Proof.
  intros Hsteady HP a. split; [ exact (steady_adds _ _ (Hsteady a)) | ].
  apply HP.
Qed.

Lemma fold_left_makes {A : Type} (f : acc -> A -> acc)
    (ids : A -> list code_id) (P : A -> acc -> Prop) (l : list A) :
  (∀ x : A, x ∈ l -> makes (ids x) (fun a => f a x) (P x)) ->
  makes (flat_map ids l) (fun a => List.fold_left f l a)
    (fun a' => ∀ x : A, x ∈ l -> P x a').
Proof.
  induction l as [ | x l IH]; intros H a; simpl.
  - split; [ exact (steady_adds _ _ (steady_refl a)) | intros a' _ y [] ].
  - destruct (H x (or_introl eq_refl) a) as [Hadds HP].
    destruct (IH (fun y Hy => H y (or_intror Hy)) (f a x)) as [Hadds' HP'].
    split; [ exact (adds_trans _ _ _ _ _ Hadds Hadds') | ].
    intros a' Hle'' y [<- | Hy].
    + apply HP. exact (acc_le_trans _ _ _ (proj1 Hadds') Hle'').
    + exact (HP' a' Hle'' y Hy).
Qed.

(* The facts of a list are those of the graph. *)
Lemma facts_alias (l : list fact) (to_ from : node) :
  Fact_alias to_ from ∈ l ->
  g_alias (graph_of_facts (facts_of_list l)) to_ from.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_alias to_ from).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma facts_use (l : list fact) (to_ from : node) :
  Fact_use to_ from ∈ l -> g_use (graph_of_facts (facts_of_list l)) to_ from.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_use to_ from).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma facts_accessor (l : list fact) (to_ : node) (f : field) (base : node) :
  Fact_accessor to_ f base ∈ l ->
  g_accessor (graph_of_facts (facts_of_list l)) to_ f base.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_accessor to_ f base).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma facts_constructor (l : list fact) (base : node) (f : field)
    (from : node) :
  Fact_constructor base f from ∈ l ->
  g_constructor (graph_of_facts (facts_of_list l)) base f from.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_constructor base f from).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma facts_argument (l : list fact) (from : node) (i : nat) (base : node) :
  Fact_argument from i base ∈ l ->
  g_argument (graph_of_facts (facts_of_list l)) from i base.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_argument from i base).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma facts_parameter (l : list fact) (base : node) (i : nat) (to_ : node) :
  Fact_parameter base i to_ ∈ l ->
  g_parameter (graph_of_facts (facts_of_list l)) base i to_.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_parameter base i to_).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma facts_propagate (l : list fact) (if_used to_ from : node) :
  Fact_propagate if_used to_ from ∈ l ->
  g_propagate (graph_of_facts (facts_of_list l)) if_used to_ from.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_propagate if_used to_ from).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma facts_any_usage (l : list fact) (x : node) :
  Fact_any_usage x ∈ l -> g_any_usage (graph_of_facts (facts_of_list l)) x.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_any_usage x).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma facts_any_source (l : list fact) (x : node) :
  Fact_any_source x ∈ l -> g_any_source (graph_of_facts (facts_of_list l)) x.
Proof.
  intros H. cbn. apply in_flat_map. exists (Fact_any_source x).
  split; [ exact H | left; reflexivity ].
Qed.

(* The primitive steps of the accumulator. *)
Lemma add_facts_steady (l : list fact) (a : acc) : steady a (add_facts l a).
Proof.
  split; [ | reflexivity ]. repeat split; simpl; auto.
  intros f H. apply in_or_app. right. exact H.
Qed.

Lemma add_facts_later (l : list fact) (a a' : acc) (f : fact) :
  acc_le (add_facts l a) a' -> f ∈ l -> f ∈ acc_facts a'.
Proof.
  intros [Hle _] Hin. apply Hle. simpl. apply in_or_app. left. exact Hin.
Qed.

Lemma fresh_steady (a : acc) : steady a (snd (fresh a)).
Proof. split; [ repeat split; simpl; auto | reflexivity ]. Qed.

Lemma fresh_list_steady (n : nat) (a : acc) :
  steady a (snd (fresh_list n a)).
Proof.
  revert a. induction n as [ | n IH]; intros a; [ apply steady_refl | ].
  change (fresh_list (S n) a)
    with (let '(x, a) := fresh a in
          let '(xs, a) := fresh_list n a in
          (x :: xs, a)).
  pose proof (fresh_steady a) as Hle. destruct (fresh a) as [x a1].
  specialize (IH a1). destruct (fresh_list n a1) as [xs a2]. simpl in *.
  exact (steady_trans _ _ _ Hle IH).
Qed.

Lemma fresh_list_length (n : nat) (a : acc) :
  length (fst (fresh_list n a)) = n.
Proof.
  revert a. induction n as [ | n IH]; intros a; [ reflexivity | ].
  change (fresh_list (S n) a)
    with (let '(x, a) := fresh a in
          let '(xs, a) := fresh_list n a in
          (x :: xs, a)).
  destruct (fresh a) as [x a1]. specialize (IH a1).
  destruct (fresh_list n a1) as [xs a2]. simpl in *. f_equal. exact IH.
Qed.

Lemma note_simple_steady (s : simple) (a : acc) : steady a (note_simple s a).
Proof. destruct s; simpl; split; try reflexivity; repeat split; simpl; auto.
Qed.

Lemma note_simple_noted (s : simple) (a a' : acc) :
  acc_le (note_simple s a) a' -> simple_noted (acc_noted a') s.
Proof.
  intros (_ & Hle & _). destruct s as [x | sym | c]; simpl; auto.
  apply Hle. left. reflexivity.
Qed.

Lemma note_simples_steady (l : list simple) (a : acc) :
  steady a (note_simples l a).
Proof.
  unfold note_simples. revert a. induction l as [ | s l IH]; intros a; simpl.
  - apply steady_refl.
  - exact (steady_trans _ _ _ (note_simple_steady s a) (IH _)).
Qed.

Lemma note_simples_noted (l : list simple) (a a' : acc) :
  acc_le (note_simples l a) a' -> simples_noted (acc_noted a') l.
Proof.
  unfold note_simples. revert a. induction l as [ | s l IH]; intros a Hle.
  - intros s [].
  - intros s' [<- | Hin]; simpl in Hle.
    + apply (note_simple_noted s a).
      exact (acc_le_trans _ _ _ (proj1 (note_simples_steady l _)) Hle).
    + exact (IH _ Hle s' Hin).
Qed.

Lemma cond_any_usage_steady (env : traverse_env) (s : simple) (a : acc) :
  steady a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage.
  destruct (env_current_code_id env);
    exact (steady_trans _ _ _ (note_simple_steady s a) (add_facts_steady _ _)).
Qed.

Lemma cond_alias_steady (env : traverse_env) (from to_ : node) (a : acc) :
  steady a (cond_alias env from to_ a).
Proof.
  unfold cond_alias. destruct (env_current_code_id env); apply add_facts_steady.
Qed.

(* [cond_alias] makes the edge of its code. *)
Lemma cond_alias_later (env : traverse_env) (from to_ : node) (a a' : acc) :
  acc_le (cond_alias env from to_ a) a' ->
  cond_alias_edge (acc_graph a') (env_current_code_id env) to_ from.
Proof.
  unfold cond_alias, cond_alias_edge.
  destruct (env_current_code_id env); intros Hle.
  - apply facts_propagate. apply (add_facts_later _ _ _ _ Hle). left.
    reflexivity.
  - apply facts_alias. apply (add_facts_later _ _ _ _ Hle). left.
    reflexivity.
Qed.

Lemma push_closure_dep_steady (d : closure_dep) (a : acc) :
  steady a (push_closure_dep d a).
Proof. split; [ repeat split; simpl; auto | reflexivity ]. Qed.

Lemma push_apply_dep_steady (d : apply_dep) (a : acc) :
  steady a (push_apply_dep d a).
Proof. split; [ repeat split; simpl; auto | reflexivity ]. Qed.

Lemma find_cont_steady (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  steady a (snd (find_cont env k n a)).
Proof.
  unfold find_cont. destruct (lookup k (env_conts env));
    [ apply steady_refl | ].
  pose proof (fresh_list_steady n a) as Hle.
  destruct (fresh_list n a) as [params a'] eqn:E. simpl in *.
  exact (steady_trans _ _ _ Hle (add_facts_steady _ _)).
Qed.

Lemma find_cont_bound (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) (params : list node) :
  lookup k (env_conts env) = Some params -> find_cont env k n a = (params, a).
Proof. intros H. unfold find_cont. rewrite H. reflexivity. Qed.

(* A position of a list and its element are among the [indexed]
   elements. *)
Lemma combine_seq_nth {A : Type} (l : list A) (k i : nat) (x : A) :
  nth_error l i = Some x -> (k + i, x) ∈ combine (seq k (length l)) l.
Proof.
  revert k i.
  induction l as [ | y l IH]; intros k i H; [ destruct i; discriminate | ].
  destruct i as [ | i]; simpl in H.
  - injection H as ->. left. f_equal. lia.
  - right. replace (k + S i) with (S k + i) by lia. apply IH, H.
Qed.

Lemma indexed_nth {A : Type} (l : list A) (i : nat) (x : A) :
  nth_error l i = Some x -> (i, x) ∈ indexed l.
Proof. intros H. exact (combine_seq_nth l 0 i x H). Qed.

(* ======================================================================== *)
(* The facts of each construct                                              *)
(* ======================================================================== *)

Lemma block_deps_makes (base : node) (fields : list simple) :
  makes [] (block_deps base fields)
    (fun a' => block_edges (acc_graph a') (acc_noted a') base fields).
Proof.
  apply makes_steady.
  - intros a. exact (steady_trans _ _ _ (note_simples_steady fields a)
                       (add_facts_steady _ _)).
  - intros a a' Hle. unfold block_deps in Hle.
    split; [ | split; [ | split ] ].
    + intros i s Hnth. apply facts_constructor.
      apply (add_facts_later _ _ _ _ Hle). apply in_or_app. left.
      apply in_map_iff. exists (i, s).
      split; [ reflexivity | exact (indexed_nth fields i s Hnth) ].
    + apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
      apply in_or_app. right. left. reflexivity.
    + apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
      apply in_or_app. right. right. left. reflexivity.
    + apply (note_simples_noted fields a a').
      exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle).
Qed.

Lemma alias_constants_makes (base : node) :
  makes [] (add_fact (Fact_alias base Node_all_constants))
    (fun a' => g_alias (acc_graph a') base Node_all_constants).
Proof.
  apply makes_steady; [ intros a; apply add_facts_steady | ].
  intros a a' Hle. apply facts_alias. apply (add_facts_later _ _ _ _ Hle).
  left. reflexivity.
Qed.

Lemma boxed_number_makes (s : symbol) (contents : simple) :
  makes [] (add_facts (boxed_number_facts (Node_symbol s) contents))
    (fun a' => boxed_edges (acc_graph a') s contents).
Proof.
  apply makes_steady; [ intros a; apply add_facts_steady | ].
  intros a a' Hle. unfold boxed_edges.
  unfold boxed_number_facts in Hle.
  destruct (boxed_number_kind contents) as [bn | ].
  - apply facts_constructor. apply (add_facts_later _ _ _ _ Hle). left.
    reflexivity.
  - split.
    + apply facts_alias. apply (add_facts_later _ _ _ _ Hle). left.
      reflexivity.
    + intros x ->. apply facts_use. apply (add_facts_later _ _ _ _ Hle).
      right. left. reflexivity.
Qed.

(* A constant of a kind of number is a boxed number of that kind. *)
Lemma boxed_number_kind_of_kind (bn : boxable_number) (c : const) :
  is_number_of_kind bn c = true -> boxed_number_kind (Const c) = Some bn.
Proof. destruct bn, c; simpl; congruence. Qed.

Lemma accessor_makes (x : variable) (f : field) (s : simple) :
  makes []
    (fun a => add_fact (Fact_accessor (Node_variable x) f (node_of_simple s))
                (note_simple s a))
    (fun a' => accessor_edges (acc_graph a') (acc_noted a') x f s).
Proof.
  apply makes_steady.
  - intros a.
    exact (steady_trans _ _ _ (note_simple_steady s a) (add_facts_steady _ _)).
  - intros a a' Hle. split.
    + apply facts_accessor. apply (add_facts_later _ _ _ _ Hle). left.
      reflexivity.
    + apply (note_simple_noted s a a').
      exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle).
Qed.

(* A primitive of the default case. *)
Lemma traverse_prim_default (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  prim_classified p args = false ->
  traverse_prim env x p args a
  = add_facts
      (Fact_any_source (Node_variable x)
       :: map (fun from => Fact_use (Node_variable x) from)
            (free_names_of_simples args))
      (if prim_has_arbitrary_effects p then cond_any_usage env (Var x) a
       else a).
Proof.
  unfold prim_classified, traverse_prim.
  destruct p as [u | b | v | h | r];
    [ destruct u | destruct b | destruct v | destruct h | destruct r ];
    repeat (match goal with b : bool |- _ => destruct b end);
    destruct args as [ | s1 [ | s2 args]]; intros H;
    try discriminate H; reflexivity.
Qed.

(* The facts of a primitive, by the cases of [traverse_prim]. *)
Lemma makes_and_const (ids : list code_id) (step : acc -> acc) (Q : Prop)
    (P : acc -> Prop) :
  Q -> makes ids step P -> makes ids step (fun a' => Q ∧ P a').
Proof.
  intros HQ H a. destruct (H a) as [Hadds HP].
  split; [ exact Hadds | intros a' Hle; exact (conj HQ (HP a' Hle)) ].
Qed.

(* The facts of a primitive, of the heap or not. *)
Lemma traverse_prim_makes_any (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) :
  makes [] (traverse_prim env x p args)
    (fun a' =>
       if prim_classified p args
       then classified_prim_edges (acc_graph a') (acc_noted a') x p args
       else default_prim_edges (acc_graph a') x args).
Proof.
  destruct (prim_classified p args) eqn:Hcl.
  - destruct p as [p' | b | v | h | r].
    + destruct p' as [kind mut fld | [ | ] | | | ak | | ok | ik op | src dst
                     | | bn | bn | | | r];
        destruct args as [ | s1 [ | s2 args]]; simpl in Hcl;
        try discriminate Hcl.
      * (* [Block_load] *)
        apply makes_steady.
        -- intros a. unfold traverse_prim.
           pose proof (steady_trans _ _ _ (note_simple_steady s1 a)
                         (add_facts_steady
                            [Fact_accessor (Node_variable x)
                               (Field_block (Z.to_nat fld) Value)
                               (node_of_simple s1)] _)) as Hs.
           destruct mut; [ exact (steady_trans _ _ _ Hs (add_facts_steady _ _))
                         | exact Hs | exact Hs ].
        -- intros a a' Hle. unfold traverse_prim in Hle.
           apply (proj2 (accessor_makes x (Field_block (Z.to_nat fld) Value)
                           s1 a)).
           destruct mut; [ | exact Hle | exact Hle ].
           exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle).
      * exact (accessor_makes x Field_is_int s1).
      * exact (accessor_makes x Field_get_tag s1).
      * exact (accessor_makes x (Field_boxed_number bn) s1).
      * (* [Box_number] *)
        apply makes_steady.
        -- intros a. exact (steady_trans _ _ _ (note_simple_steady s1 a)
                              (add_facts_steady _ _)).
        -- intros a a' Hle. split.
           ++ apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
              left. reflexivity.
           ++ apply (note_simple_noted s1 a a').
              exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle).
    + discriminate Hcl.
    + destruct v; [ exact (block_deps_makes (Node_variable x) args)
                  | discriminate Hcl ].
    + destruct h; try discriminate Hcl.
      exact (block_deps_makes (Node_variable x) args).
    + (* [Local_allocation]: no fact. *)
      destruct r; try discriminate Hcl.
      apply makes_steady; [ intros a; apply steady_refl | ].
      intros a a' _. destruct args; exact I.
  - apply (makes_ext [] (fun a =>
             add_facts
               (Fact_any_source (Node_variable x)
                :: map (fun from => Fact_use (Node_variable x) from)
                     (free_names_of_simples args))
               (if prim_has_arbitrary_effects p
                then cond_any_usage env (Var x) a else a))).
    { intros a. symmetry. exact (traverse_prim_default env x p args a Hcl). }
    apply makes_steady.
    + intros a. refine (steady_trans _ _ _ _ (add_facts_steady _ _)).
      destruct (prim_has_arbitrary_effects p);
        [ apply cond_any_usage_steady | apply steady_refl ].
    + intros a a' Hle. split.
      * apply facts_any_source. apply (add_facts_later _ _ _ _ Hle). left.
        reflexivity.
      * intros from Hfrom. apply facts_use. apply (add_facts_later _ _ _ _ Hle).
        right. apply in_map_iff. exists from.
        split; [ reflexivity | exact Hfrom ].
Qed.

(* [cond_any_usage] makes the edge of its code. *)
Lemma cond_any_usage_later (env : traverse_env) (s : simple) (a a' : acc) :
  acc_le (cond_any_usage env s a) a' ->
  cond_use_edge (acc_graph a') (env_current_code_id env) (node_of_simple s).
Proof.
  unfold cond_any_usage, cond_use_edge.
  destruct (env_current_code_id env); intros Hle.
  - apply facts_use. apply (add_facts_later _ _ _ _ Hle). left. reflexivity.
  - apply facts_any_usage. apply (add_facts_later _ _ _ _ Hle). left.
    reflexivity.
Qed.

(* A primitive that [traverse_prim] reads in a way of its own has no
   arbitrary effects. *)
Lemma classified_no_effects (p : prim) (args : list simple) :
  prim_classified p args = true -> prim_has_arbitrary_effects p = false.
Proof.
  destruct p as [u | b | v | h | r];
    [ destruct u | destruct b | destruct v | destruct h | destruct r ];
    repeat (match goal with b : bool |- _ => destruct b end);
    destruct args as [ | s1 [ | s2 args]]; intros H;
    try discriminate H; reflexivity.
Qed.

(* The use of the variable of a primitive with arbitrary effects by the
   code it is in, and any source for the variable of a load of a mutable
   field. *)
Lemma traverse_prim_side (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) :
  makes [] (traverse_prim env x p args)
    (fun a' =>
       (prim_has_arbitrary_effects p = true ->
        cond_use_edge (acc_graph a') (env_current_code_id env)
          (Node_variable x))
       ∧ (prim_loads_mutable p = true ->
          g_any_source (acc_graph a') (Node_variable x))).
Proof.
  intros a. split; [ exact (proj1 (traverse_prim_makes_any env x p args a)) | ].
  intros a' Hle.
  destruct (prim_classified p args) eqn:Hcl.
  - split; [ intros Heff; rewrite (classified_no_effects p args Hcl) in Heff;
             discriminate Heff | ].
    intros Hmut.
    destruct p as [p' | | | | ]; try discriminate Hmut.
    destruct p' as [kind mut fld | | | | | | | | | | | | | | ];
      try discriminate Hmut.
    destruct mut; try discriminate Hmut.
    destruct args as [ | s1 [ | s2 args]]; try discriminate Hcl.
    apply facts_any_source. apply (add_facts_later _ _ _ _ Hle). left.
    reflexivity.
  - rewrite (traverse_prim_default env x p args a Hcl) in Hle. split.
    + intros Heff. rewrite Heff in Hle.
      exact (cond_any_usage_later env (Var x) a a'
               (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle)).
    + intros _. apply facts_any_source. apply (add_facts_later _ _ _ _ Hle).
      left. reflexivity.
Qed.

Lemma traverse_prim_makes (heap_prims : Prop) (env : traverse_env)
    (x : variable) (p : prim) (args : list simple) :
  prim_of_heap p = false ∨ heap_prims ->
  makes [] (traverse_prim env x p args)
    (fun a' =>
       prim_edges (acc_graph a') (acc_noted a') heap_prims
         (env_current_code_id env) x p args).
Proof.
  intros Hoff a. unfold prim_edges.
  destruct (traverse_prim_side env x p args a) as [Hadds Hside].
  destruct (traverse_prim_makes_any env x p args a) as [_ Hany].
  split; [ exact Hadds | intros a' Hle ].
  split; [ | exact (Hany a' Hle) ].
  split; [ exact Hoff | exact (Hside a' Hle) ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Sets of closures                                                         *)
(* ------------------------------------------------------------------------ *)

(* The closures of a [Closure_of_set]: one for each function slot, the
   one at the slot of the [Let] being its variable. *)
Lemma closures_of_set_spec (x : variable) (slot : function_slot)
    (others : list (function_slot * variable))
    (fds : list (function_slot * code_id)) (a : acc) :
  steady a (snd (closures_of_set x slot others fds a))
  ∧ map (fun '(f, _, id) => (f, id))
      (fst (closures_of_set x slot others fds a))
    = fds
  ∧ ∀ (f : function_slot) (name : node) (id : code_id),
      (f, name, id) ∈ fst (closures_of_set x slot others fds a) ->
      f = slot -> name = Node_variable x.
Proof.
  revert a. induction fds as [ | [f id] fds IH]; intros a.
  - split; [ apply steady_refl | split; [ reflexivity | intros ? ? ? [] ] ].
  - simpl.
    set (named := if Pos.eqb f slot then (Node_variable x, a)
                  else match lookup f others with
                       | Some y => (Node_variable y, a)
                       | None => fresh a
                       end).
    assert (Hname : steady a (snd named)
                    ∧ (f = slot -> fst named = Node_variable x)).
    { unfold named. destruct (Pos.eqb_spec f slot) as [-> | Hne].
      - split; [ apply steady_refl | reflexivity ].
      - split; [ | intros Heq; contradiction ].
        destruct (lookup f others);
          [ apply steady_refl | apply fresh_steady ]. }
    destruct named as [name a1].
    destruct Hname as [Hs1 Hn1]. simpl in Hs1, Hn1.
    destruct (IH a1) as (Hs2 & Hmap & Hslot).
    destruct (closures_of_set x slot others fds a1) as [rest a2].
    simpl in *.
    split; [ exact (steady_trans _ _ _ Hs1 Hs2) | ].
    split; [ f_equal; exact Hmap | ].
    intros f' name' id' [[= <- <- <-] | Hin] Heq; [ exact (Hn1 Heq) | ].
    exact (Hslot _ _ _ Hin Heq).
Qed.

(* The node of each closure, as the list of the closures has it. *)
Definition sites_of (x : variable)
    (closures : list (function_slot * node * code_id)) (f : function_slot)
    : node :=
  match lookup f (map (fun '(f, name, _) => (f, name)) closures) with
  | Some name => name
  | None => Node_variable x
  end.

Lemma sites_of_lookup (x : variable)
    (closures : list (function_slot * node * code_id))
    (fds : list (function_slot * code_id)) (f : function_slot)
    (id : code_id) :
  map (fun '(f, _, id) => (f, id)) closures = fds ->
  lookup f fds = Some id ->
  (f, sites_of x closures f, id) ∈ closures.
Proof.
  unfold sites_of. revert fds.
  induction closures as [ | [[f' name] id'] closures IH]; intros fds Hmap;
    simpl in Hmap; subst fds; simpl; [ discriminate | ].
  destruct (Pos.eqb_spec f f') as [-> | Hne].
  - intros [= ->]. left. reflexivity.
  - intros Hl. right. exact (IH _ eq_refl Hl).
Qed.

Lemma sites_of_slot (x : variable)
    (closures : list (function_slot * node * code_id))
    (slot : function_slot) :
  (∀ (f : function_slot) (name : node) (id : code_id),
     (f, name, id) ∈ closures -> f = slot -> name = Node_variable x) ->
  sites_of x closures slot = Node_variable x.
Proof.
  intros H. unfold sites_of.
  destruct (lookup slot (map (fun '(f, name, _) => (f, name)) closures))
    as [name | ] eqn:Hl; [ | reflexivity ].
  apply lookup_in, in_map_iff in Hl as ([[f name'] id] & [= <- <-] & Hin).
  exact (H _ _ _ Hin eq_refl).
Qed.

(* The facts of each closure of a set. *)
Definition closure_facts (env : traverse_env)
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (base : node)
    (code : option code_id) (a' : acc) : Prop :=
  (∀ id : code_id,
     code = Some id ->
     acc_deps a'
       {| cld_let_bound_name_of_the_closure := base;
          cld_closure_code_id := id |})
  ∧ (∀ (f : function_slot) (closure : node),
       (f, closure) ∈ function_slots ->
       g_constructor (acc_graph a') base (Field_function_slot f) closure)
  ∧ (∀ (w : value_slot) (s : simple),
       lookup w value_slots = Some s ->
       should_keep_value_slot env w = true ->
       g_constructor (acc_graph a') base (Field_value_slot w)
         (node_of_simple s)
       ∧ simple_noted (acc_noted a') s).

Lemma record_set_makes (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) :
  makes [] (record_set_of_closures_deps env closures function_slots value_slots)
    (fun a' =>
       ∀ (base : node) (code : option code_id),
         (base, code) ∈ closures ->
         closure_facts env function_slots value_slots base code a').
Proof.
  unfold record_set_of_closures_deps.
  set (kept := filter (fun '(w, _) => should_keep_value_slot env w)
                 (compact value_slots)).
  apply (makes_perm (flat_map (fun _ => []) closures));
    [ clear; induction closures as [ | ? ? IH]; [ reflexivity | exact IH ]
    | ].
  eapply makes_weaken;
    [ | apply (fold_left_makes _ (fun _ => [])
                 (fun x a' =>
                    closure_facts env function_slots value_slots (fst x)
                      (snd x) a')) ].
  { intros a' H base code Hin. exact (H _ Hin). }
  intros [base code] _. apply makes_steady.
  - intros a. refine (steady_trans _ _ _ _
      (steady_trans _ _ _ (note_simples_steady _ _) (add_facts_steady _ _))).
    destruct code; [ apply push_closure_dep_steady | apply steady_refl ].
  - intros a a' Hle. simpl. split; [ | split ].
    + intros id [= ->]. apply Hle. apply (proj1 (proj2 (proj2
        (proj1 (steady_trans _ _ _ (note_simples_steady _ _)
                  (add_facts_steady _ _)))))).
      left. reflexivity.
    + intros f closure Hin. apply facts_constructor.
      apply (add_facts_later _ _ _ _ Hle). apply in_or_app. right.
      apply in_map_iff. exists (f, closure). auto.
    + intros w s Hl Hkeep.
      assert (Hin : (w, s) ∈ kept).
      { apply filter_In. split; [ apply in_compact, Hl | exact Hkeep ]. }
      split.
      * apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
        apply in_or_app. left. apply in_map_iff. exists (w, s). auto.
      * eapply note_simples_noted;
          [ exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle)
          | apply in_map_iff; exists (w, s); auto ].
Qed.

Lemma closure_of_set_makes (env : traverse_env) (x : variable)
    (slot : function_slot) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) :
  makes [] (traverse_closure_of_set env x slot function_decls value_slots)
    (fun a' =>
       set_edges (acc_graph a') (acc_noted a') (acc_deps a')
         (should_keep_value_slot env) x slot function_decls value_slots).
Proof.
  intros a. unfold traverse_closure_of_set.
  destruct (among slot (map fst (compact function_decls))) eqn:Hslot.
  - destruct (closures_of_set_spec x slot (set_closures_of env x)
                (compact function_decls) a)
      as (Hs & Hmap & Hname).
    destruct (closures_of_set x slot (set_closures_of env x)
                (compact function_decls) a)
      as [closures a1].
    simpl in Hs, Hmap, Hname.
    destruct (record_set_makes env
                (map (fun '(_, name, id) => (name, Some id)) closures)
                (map (fun '(f, name, _) => (f, name)) closures)
                value_slots a1) as [Hadds HP].
    split; [ exact (adds_trans [] [] _ _ _ (steady_adds _ _ Hs) Hadds) | ].
    intros a' Hle id0 _. exists (sites_of x closures).
    split; [ exact (sites_of_slot x closures slot Hname) | ].
    intros f id Hl.
    assert (Hin : (f, sites_of x closures f, id) ∈ closures).
    { apply (sites_of_lookup x closures (compact function_decls)); auto.
      rewrite lookup_compact. exact Hl. }
    destruct (HP a' Hle (sites_of x closures f) (Some id))
      as (Hdep & Hfs & Hvs).
    { apply in_map_iff. exists (f, sites_of x closures f, id). auto. }
    split; [ exact (Hdep id eq_refl) | split; [ | exact Hvs ] ].
    intros g id' Hg. apply Hfs. apply in_map_iff.
    exists (g, sites_of x closures g, id'). split; [ reflexivity | ].
    apply (sites_of_lookup x closures (compact function_decls)); auto.
    rewrite lookup_compact. exact Hg.
  - split; [ exact (steady_adds _ _ (steady_refl a)) | ].
    intros a' _ id0 Hl0. exfalso.
    assert (Hin : (slot, id0) ∈ compact function_decls)
      by (apply in_compact; exact Hl0).
    apply (in_map fst) in Hin. simpl in Hin.
    apply among_In in Hin.
    pose proof (eq_trans (eq_sym Hin) Hslot) as Hfalse. discriminate Hfalse.
Qed.

(* The facts of a [Let] of the fragment. *)
Lemma traverse_named_makes (env : traverse_env) (slots_read : Prop)
    (hb : bool) (x : variable) (defining_expr : named) :
  named_in_flow_fragment hb defining_expr = true ->
  (named_projection_free defining_expr = false -> slots_read) ->
  (∀ w : value_slot,
     w ∈ projected_value_slots_of_named defining_expr ->
     should_keep_value_slot env w = true) ->
  makes [] (traverse_named env x defining_expr)
    (fun a' =>
       named_edges (acc_graph a') (acc_noted a') (acc_deps a')
         (should_keep_value_slot env) slots_read (if hb then True else False)
         (env_current_code_id env) x defining_expr).
Proof.
  destruct defining_expr as [s | p args | | slot decls vs | | from w closure
                            | move_from move_to closure];
    simpl; intros Hfrag Hslots Hkeep; try discriminate Hfrag.
  - apply makes_steady.
    + intros a.
      exact (steady_trans _ _ _ (note_simple_steady s a) (add_facts_steady _
        _)).
    + intros a a' Hle. split.
      * apply facts_alias. apply (add_facts_later _ _ _ _ Hle). left.
        reflexivity.
      * apply (note_simple_noted s a a').
        exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle).
  - apply traverse_prim_makes.
    destruct hb; [ right; exact I | left ].
    destruct (prim_of_heap p); [ discriminate Hfrag | reflexivity ].
  - apply closure_of_set_makes.
  - eapply makes_weaken; [ | exact (accessor_makes x (Field_value_slot w)
                                       closure) ].
    intros a' H. split; [ exact H | split;
      [ apply Hkeep; left; reflexivity | exact (Hslots eq_refl) ] ].
  - eapply makes_weaken; [ | exact (accessor_makes x
                                      (Field_function_slot move_to) closure) ].
    intros a' H. split; [ exact H | exact (Hslots eq_refl) ].
Qed.

(* A static closure of a set. *)
Lemma static_set_makes (env : traverse_env) (symbols : list symbol)
    (s : symbol) (slot : function_slot)
    (closure_symbols : list (function_slot * symbol))
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) :
  is_bound slot function_decls
  && forallb (fun '(_, s) => among s symbols) closure_symbols = true ->
  makes []
    (traverse_static_binding env
       (Bind_symbol s
          (Static_const_closure_of_set slot closure_symbols function_decls
             value_slots)))
    (fun a' =>
       static_set_edges (acc_graph a') (acc_noted a') (acc_deps a')
         (should_keep_value_slot env) symbols s slot closure_symbols
         function_decls value_slots).
Proof.
  intros Hfrag. apply andb_true_iff in Hfrag as [Hslot Hsyms]. simpl.
  eapply makes_weaken; [ | apply record_set_makes ].
  intros a' H.
  destruct (H (Node_symbol s) (lookup slot function_decls) (or_introl eq_refl))
    as (Hdep & Hfs & Hvs).
  split; [ exact Hslot | ].
  split; [ exact Hdep | split; [ | exact Hvs ] ].
  intros g s' Hl. split.
  - apply Hfs. apply in_map_iff. exists (g, s').
    split; [ reflexivity | apply in_compact; exact Hl ].
  - rewrite forallb_forall in Hsyms.
    exact (Hsyms (g, s') (lookup_in _ _ _ Hl)).
Qed.

(* An [Apply_cont] of a continuation in scope aliases its parameters to
   the arguments. *)
Lemma apply_cont_deps_makes (env : traverse_env) (ac : apply_cont) :
  makes [] (apply_cont_deps env ac)
    (fun a' => apply_cont_edges (acc_graph a') (acc_noted a')
                 (env_conts env) ac).
Proof.
  apply makes_steady.
  - intros a. unfold apply_cont_deps.
    pose proof (find_cont_steady env (ac_continuation ac)
                  (length (ac_args ac)) a) as H0.
    destruct (find_cont env (ac_continuation ac) (length (ac_args ac)) a)
      as [params a1].
    exact (steady_trans _ _ _ H0 (steady_trans _ _ _ (note_simples_steady _ _)
                                    (add_facts_steady _ _))).
  - intros a a' Hle. unfold apply_cont_deps in Hle.
    destruct (lookup (ac_continuation ac) (env_conts env)) as [params | ]
      eqn:Hlookup.
    + rewrite (find_cont_bound _ _ _ _ _ Hlookup) in Hle. split.
      * intros params' Hparams p s Hin.
        rewrite Hlookup in Hparams. injection Hparams as <-.
        apply facts_alias. apply (add_facts_later _ _ _ _ Hle).
        apply in_map_iff. exists (p, s). split; [ reflexivity | exact Hin ].
      * eapply note_simples_noted.
        exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle).
    + destruct (find_cont env (ac_continuation ac) (length (ac_args ac)) a)
        as [params a1].
      split.
      * intros params' Hparams. rewrite Hlookup in Hparams. discriminate.
      * eapply note_simples_noted.
        exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle).
Qed.

Lemma traverse_switch_makes (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  makes [] (traverse_switch env scrutinee arms)
    (fun a' =>
       every
         (fun '(_, ac) =>
            apply_cont_edges (acc_graph a') (acc_noted a') (env_conts env) ac)
         (Keyed.compact Z.eqb arms)).
Proof.
  assert (Hfold :
    makes (flat_map (fun _ => []) (Keyed.compact Z.eqb arms))
      (fun a =>
         List.fold_left (fun a '(_, ac) => apply_cont_deps env ac a)
           (Keyed.compact Z.eqb arms) a)
      (fun a' =>
         ∀ arm, arm ∈ Keyed.compact Z.eqb arms ->
           (fun '(_, ac) =>
              apply_cont_edges (acc_graph a') (acc_noted a') (env_conts env)
                ac) arm)).
  { apply fold_left_makes. intros [d ac] _. apply apply_cont_deps_makes. }
  apply (makes_perm _ _ _ _ (Permutation_refl _)) in Hfold.
  pose proof (makes_compose _ _ _ _ _ _
                (makes_steady _ (fun _ => True)
                   (cond_any_usage_steady env scrutinee) (fun _ _ _ => I))
                Hfold) as Hboth.
  apply (makes_perm _ []) in Hboth;
    [ | clear; induction (Keyed.compact Z.eqb arms) as [ | ? ? IH];
        [ reflexivity | exact IH ] ].
  apply (makes_weaken _ _ _ _
           (fun a' (H : True ∧ _) => proj2 (every_in _ _ _) (proj2 H)))
    in Hboth.
  exact Hboth.
Qed.

Lemma exn_handler_deps_steady (env : traverse_env)
    (h : continuation_handler) (a : acc) :
  steady a (exn_handler_deps env h a).
Proof.
  destruct h as [[ | p params] metadata handler]; simpl;
    [ apply steady_refl | ].
  destruct (hm_is_exn_handler metadata); [ | apply steady_refl ].
  exact (steady_trans _ _ _ (add_facts_steady _ _)
           (cond_any_usage_steady _ _ _)).
Qed.

Lemma add_rec_conts_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) :
  env_conts (add_rec_conts env handlers)
  = rec_conts (env_conts env) handlers.
Proof.
  unfold rec_conts.
  induction handlers as [ | [k h] handlers IH]; [ reflexivity | ].
  simpl. f_equal. exact IH.
Qed.

Lemma add_rec_conts_fields (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) :
  env_current_code_id (add_rec_conts env handlers) = env_current_code_id env
  ∧ env_value_slots_to_keep (add_rec_conts env handlers)
    = env_value_slots_to_keep env.
Proof.
  induction handlers as [ | [k h] handlers IH]; [ auto | ].
  simpl. exact IH.
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

(* A step that returns a node and an accumulator. *)
Lemma steady_snd {B : Type} (f : acc -> B * acc) (a : acc) (b : B)
    (a' : acc) :
  (∀ a, steady a (snd (f a))) -> f a = (b, a') -> steady a a'.
Proof. intros H E. specialize (H a). rewrite E in H. exact H. Qed.

(* The witness of a direct call, which takes the arguments and gives the
   results and the exception. *)
Lemma known_widget_spec (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  steady a (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ ∀ a' : acc,
      acc_le (snd (make_known_arity_apply_widget env ap returns exn a)) a' ->
      ∃ witness called_node : node,
        (∀ (i : nat) (s : simple),
           nth_error (ap_args ap) i = Some s ->
           g_argument (acc_graph a') (node_of_simple s) i witness)
        ∧ (∀ (i : nat) (r : node),
             nth_error returns i = Some r ->
             g_accessor (acc_graph a') r (Field_return_of_call (Normal i))
               witness)
        ∧ g_accessor (acc_graph a') exn (Field_return_of_call Exn) witness
        ∧ g_accessor (acc_graph a') called_node
            Field_code_id_of_call_witness witness
        ∧ g_any_usage (acc_graph a') called_node
        ∧ cond_alias_edge (acc_graph a') (env_current_code_id env) witness
            (fst (make_known_arity_apply_widget env ap returns exn a))
        ∧ simples_noted (acc_noted a') (ap_args ap).
Proof.
  unfold make_known_arity_apply_widget.
  destruct (fresh a) as [witness a1] eqn:E1.
  pose proof (steady_snd fresh a witness a1 fresh_steady E1) as S1.
  destruct (fresh a1) as [called a2] eqn:E2.
  pose proof (steady_snd fresh a1 called a2 fresh_steady E2) as S2.
  destruct (fresh a2) as [apply_node a3] eqn:E3.
  pose proof (steady_snd fresh a2 apply_node a3 fresh_steady E3) as S3.
  set (facts_list :=
         map (fun '(i, s) => Fact_argument (node_of_simple s) i witness)
           (indexed (ap_args ap))
         ++ map (fun '(i, r) =>
                   Fact_accessor r (Field_return_of_call (Normal i)) witness)
              (indexed returns)
         ++ [Fact_accessor exn (Field_return_of_call Exn) witness;
             Fact_accessor called Field_code_id_of_call_witness witness;
             Fact_any_usage called]).
  simpl. split.
  - refine (steady_trans _ _ _ S1 (steady_trans _ _ _ S2
              (steady_trans _ _ _ S3 _))).
    refine (steady_trans _ _ _ (note_simples_steady _ _) _).
    refine (steady_trans _ _ _ (add_facts_steady _ _) _).
    apply cond_alias_steady.
  - intros a' Hle.
    pose proof (acc_le_trans _ _ _ (proj1 (cond_alias_steady _ _ _ _)) Hle)
      as Hle'.
    exists witness, called. split; [ | split; [ | split; [ | split;
      [ | split; [ | split ] ] ] ] ].
    + intros i s Hnth. apply facts_argument.
      apply (add_facts_later _ _ _ _ Hle'). apply in_or_app. left.
      apply in_map_iff. exists (i, s). split; [ reflexivity | ].
      exact (indexed_nth _ _ _ Hnth).
    + intros i r Hnth. apply facts_accessor.
      apply (add_facts_later _ _ _ _ Hle'). apply in_or_app. right.
      apply in_or_app. left. apply in_map_iff. exists (i, r).
      split; [ reflexivity | exact (indexed_nth _ _ _ Hnth) ].
    + apply facts_accessor. apply (add_facts_later _ _ _ _ Hle').
      apply in_or_app. right. apply in_or_app. right. left. reflexivity.
    + apply facts_accessor. apply (add_facts_later _ _ _ _ Hle').
      apply in_or_app. right. apply in_or_app. right. right. left.
      reflexivity.
    + apply facts_any_usage. apply (add_facts_later _ _ _ _ Hle').
      apply in_or_app. right. apply in_or_app. right. right. right. left.
      reflexivity.
    + exact (cond_alias_later _ _ _ _ _ Hle).
    + eapply note_simples_noted.
      exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle').
Qed.

Lemma add_apply_steady (env : traverse_env) (id : code_id)
    (callee call_widget : node) (b : bool) (a : acc) :
  steady a (add_apply env id callee call_widget b a).
Proof.
  unfold add_apply. destruct b; [ | apply push_apply_dep_steady ].
  destruct (fresh a) as [c1 a1] eqn:E1.
  pose proof (steady_snd fresh a c1 a1 fresh_steady E1) as S1.
  destruct (fresh a1) as [c2 a2] eqn:E2.
  pose proof (steady_snd fresh a1 c2 a2 fresh_steady E2) as S2.
  refine (steady_trans _ _ _ S1 (steady_trans _ _ _ S2 _)).
  exact (steady_trans _ _ _ (add_facts_steady _ _)
           (push_apply_dep_steady _ _)).
Qed.

(* The witnesses of a call of unknown arity. *)
Lemma unknown_apply_deps_spec (returns : list node) (exn called : node)
    (l : list (list simple * node)) (a : acc) :
  steady a (unknown_arity_apply_deps returns exn called l a)
  ∧ ∀ a' : acc,
      acc_le (unknown_arity_apply_deps returns exn called l a) a' ->
      caller_chain_edges (acc_graph a') returns exn called l.
Proof.
  revert a. induction l as [ | [first w] rest IH]; intros a.
  - split; [ apply steady_refl | intros; exact I ].
  - set (l1 := map (fun '(i, s) => Fact_argument (node_of_simple s) i w)
                 (indexed first)
               ++ [Fact_accessor exn (Field_return_of_call Exn) w;
                   Fact_accessor called Field_code_id_of_call_witness w]).
    set (a1 := add_facts l1 a).
    assert (S1 : steady a a1) by apply add_facts_steady.
    assert (H1 : ∀ a', acc_le a1 a' ->
              (∀ (k : nat) (s : simple),
                 nth_error first k = Some s ->
                 g_argument (acc_graph a') (node_of_simple s) k w)
              ∧ g_accessor (acc_graph a') exn (Field_return_of_call Exn) w
              ∧ g_accessor (acc_graph a') called
                  Field_code_id_of_call_witness w).
    { intros a' Hle. split; [ | split ].
      - intros k s Hk. apply facts_argument.
        apply (add_facts_later _ _ _ _ Hle). apply in_or_app. left.
        apply in_map_iff. exists (k, s).
        split; [ reflexivity | exact (indexed_nth _ _ _ Hk) ].
      - apply facts_accessor. apply (add_facts_later _ _ _ _ Hle).
        apply in_or_app. right. left. reflexivity.
      - apply facts_accessor. apply (add_facts_later _ _ _ _ Hle).
        apply in_or_app. right. right. left. reflexivity. }
    destruct rest as [ | [first' w'] rest'].
    + cbn [unknown_arity_apply_deps caller_chain_edges]. fold l1 a1.
      split; [ exact (steady_trans _ _ _ S1 (add_facts_steady _ _)) | ].
      intros a' Hle.
      destruct (H1 a' (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle))
        as (Ha & Hb & Hc').
      split; [ exact Ha | split; [ exact Hb | split; [ exact Hc' | ] ] ].
      split; [ | exact I ].
      intros i r Hr. apply facts_accessor. apply (add_facts_later _ _ _ _ Hle).
      apply in_map_iff. exists (i, r).
      split; [ reflexivity | exact (indexed_nth _ _ _ Hr) ].
    + cbn [unknown_arity_apply_deps caller_chain_edges]. fold l1 a1.
      destruct (fresh a1) as [v a2] eqn:E.
      pose proof (steady_snd fresh a1 v a2 fresh_steady E) as S2.
      set (a3 := add_facts
                   [Fact_accessor v (Field_return_of_call (Normal 0)) w;
                    Fact_accessor w'
                      (Field_call_witness Unknown_arity_code_pointer) v] a2).
      destruct (IH a3) as [S4 H4].
      split; [ exact (steady_trans _ _ _ S1 (steady_trans _ _ _ S2
                (steady_trans _ _ _ (add_facts_steady _ _) S4))) | ].
      intros a' Hle.
      pose proof (acc_le_trans _ _ _ (proj1 S4) Hle) as Hle3.
      pose proof (acc_le_trans _ _ _ (proj1 S2)
                    (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle3))
        as Hle1.
      destruct (H1 a' Hle1) as (Ha & Hb & Hc').
      split; [ exact Ha | split; [ exact Hb | split; [ exact Hc' | split ] ] ].
      * exists v. split; apply facts_accessor;
          apply (add_facts_later _ _ _ _ Hle3);
          [ left | right; left ]; reflexivity.
      * exact (H4 a' Hle).
Qed.

Lemma unknown_widget_spec (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  steady a (snd (make_unknown_arity_apply_widget env ap returns exn a))
  ∧ ∀ a' : acc,
      acc_le (snd (make_unknown_arity_apply_widget env ap returns exn a)) a' ->
      ∃ (ws : list node) (called_node : node),
        length ws = length (call_groups ap)
        ∧ caller_chain_edges (acc_graph a') returns exn called_node
            (combine (call_groups ap) ws)
        ∧ g_any_usage (acc_graph a') called_node
        ∧ (∀ w : node,
             hd_error ws = Some w ->
             cond_alias_edge (acc_graph a') (env_current_code_id env) w
               (fst (make_unknown_arity_apply_widget env ap returns exn a)))
        ∧ simples_noted (acc_noted a') (ap_args ap).
Proof.
  unfold make_unknown_arity_apply_widget.
  destruct (fresh a) as [called a1] eqn:E1.
  pose proof (steady_snd fresh a called a1 fresh_steady E1) as S1.
  cbv zeta. fold (call_groups ap).
  set (a2 := add_fact (Fact_any_usage called) a1).
  destruct (fresh_list (length (call_groups ap)) a2) as [ws a3] eqn:E3.
  pose proof (steady_snd _ a2 ws a3 (fresh_list_steady _) E3) as S3.
  pose proof (fresh_list_length (length (call_groups ap)) a2) as L3.
  rewrite E3 in L3. simpl in L3.
  set (a4 := note_simples (ap_args ap) a3).
  destruct (unknown_apply_deps_spec returns exn called
              (combine (call_groups ap) ws) a4) as [S5 H5].
  set (a5 := unknown_arity_apply_deps returns exn called
               (combine (call_groups ap) ws) a4) in *.
  destruct (fresh a5) as [apply_node a6] eqn:E6.
  pose proof (steady_snd fresh a5 apply_node a6 fresh_steady E6) as S6.
  assert (S : steady a a6).
  { refine (steady_trans _ _ _ S1 (steady_trans _ _ _ (add_facts_steady _ _)
              (steady_trans _ _ _ S3 (steady_trans _ _ _
                 (note_simples_steady _ _) (steady_trans _ _ _ S5 S6))))). }
  assert (Hrest : ∀ a', acc_le a6 a' ->
            length ws = length (call_groups ap)
            ∧ caller_chain_edges (acc_graph a') returns exn called
                (combine (call_groups ap) ws)
            ∧ g_any_usage (acc_graph a') called
            ∧ simples_noted (acc_noted a') (ap_args ap)).
  { intros a' Hle. split; [ exact L3 | split; [ | split ] ].
    - apply H5. exact (acc_le_trans _ _ _ (proj1 S6) Hle).
    - apply facts_any_usage.
      apply (add_facts_later [Fact_any_usage called] a1 _ _);
        [ | left; reflexivity ].
      refine (acc_le_trans _ _ _ (proj1 S3) _).
      refine (acc_le_trans _ _ _ (proj1 (note_simples_steady _ _)) _).
      exact (acc_le_trans _ _ _ (proj1 S5) (acc_le_trans _ _ _ (proj1 S6)
               Hle)).
    - apply (note_simples_noted _ a3 a').
      exact (acc_le_trans _ _ _ (proj1 S5) (acc_le_trans _ _ _ (proj1 S6)
               Hle)). }
  destruct ws as [ | first ws'].
  - simpl. split; [ exact S | ].
    intros a' Hle. destruct (Hrest a' Hle) as (L & Hch & Hu & Hn).
    exists [], called. split; [ exact L | split; [ exact Hch | ] ].
    split; [ exact Hu | split; [ intros w Hw; discriminate Hw | exact Hn ] ].
  - simpl. split; [ exact (steady_trans _ _ _ S (cond_alias_steady _ _ _ _))
                  | ].
    intros a' Hle.
    destruct (Hrest a' (acc_le_trans _ _ _ (proj1 (cond_alias_steady _ _ _ _))
                          Hle)) as (L & Hch & Hu & Hn).
    exists (first :: ws'), called.
    split; [ exact L | split; [ exact Hch | ] ].
    split; [ exact Hu | split; [ | exact Hn ] ].
    intros w [= <-]. exact (cond_alias_later _ _ _ _ _ Hle).
Qed.

(* The facts of a call of the fragment. *)
Lemma traverse_apply_makes (env : traverse_env) (codes : list (code_id * code))
    (ap : apply_expr) :
  makes [] (traverse_apply env ap)
    (fun a' =>
       apply_in_flow_fragment codes ap = true ->
       apply_edges (acc_graph a') (acc_noted a') codes
         (env_current_code_id env) (env_conts env) ap).
Proof.
  intros a. unfold traverse_apply.
  destruct (find_cont env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a)
    as [returns a1] eqn:E1.
  pose proof (steady_snd (find_cont env (ap_continuation ap)
                            (length (unarize (ap_return_arity ap)))) a
                returns a1 (find_cont_steady _ _ _) E1) as S1.
  destruct (find_cont env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) a1)
    as [exn_params a2] eqn:E2.
  pose proof (steady_snd (find_cont env (ap_exn_continuation ap)
                            (S (length (ap_exn_extra_args ap)))) a1
                exn_params a2 (find_cont_steady _ _ _) E2) as S2.
  assert (Hexn : ∃ (exn : node) (a3 : acc),
            (match exn_params with
             | [] => fresh a2
             | exn_param :: extra_params =>
               (exn_param,
                add_facts
                  (map (fun '(p, s) => Fact_alias p (node_of_simple s))
                     (combine extra_params (exn_extra_simples ap)))
                  (note_simples (exn_extra_simples ap) a2))
             end) = (exn, a3)
            ∧ steady a2 a3
            ∧ ∀ (p : node) (params : list node),
                exn_params = p :: params ->
                exn = p
                ∧ ∀ a' : acc,
                    acc_le a3 a' ->
                    simples_noted (acc_noted a') (exn_extra_simples ap)
                    ∧ ∀ (q : node) (s : simple),
                        (q, s) ∈ combine params (exn_extra_simples ap) ->
                        g_alias (acc_graph a') q (node_of_simple s)).
  { destruct exn_params as [ | p params].
    - destruct (fresh a2) as [exn a3] eqn:E3. exists exn, a3.
      split; [ reflexivity | split ];
        [ exact (steady_snd fresh a2 exn a3 fresh_steady E3)
        | intros ? ? ?; discriminate ].
    - eexists _, _. split; [ reflexivity | split ].
      + exact (steady_trans _ _ _ (note_simples_steady _ _)
                 (add_facts_steady _ _)).
      + intros p' params' [= -> ->]. split; [ reflexivity | ].
        intros a' Hle. split.
        * apply (note_simples_noted _ a2 a').
          exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle).
        * intros q s Hqs. apply facts_alias.
          apply (add_facts_later _ _ _ _ Hle). apply in_map_iff.
          exists (q, s). split; [ reflexivity | exact Hqs ]. }
  destruct Hexn as (exn & a3 & -> & S3 & Hexn).
  (* What the call kind adds. *)
  assert (Hkind : steady a3 (traverse_call_kind env ap exn returns a3)
                  ∧ ∀ a' : acc,
                      acc_le (traverse_call_kind env ap exn returns a3) a' ->
                      call_edges (acc_graph a') (env_current_code_id env) ap
                        returns exn
                      ∧ simple_noted (acc_noted a') (ap_callee ap)
                      ∧ simples_noted (acc_noted a') (ap_args ap)).
  { unfold traverse_call_kind, call_edges.
    destruct (ap_call_kind ap) as [id | | code_ids] eqn:Hk.
    - pose proof (known_widget_spec env ap returns exn
                    (note_simple (ap_callee ap) a3)) as [S4 H4].
      destruct (make_known_arity_apply_widget env ap returns exn
                  (note_simple (ap_callee ap) a3)) as [apply_node a4] eqn:E4.
      simpl in S4, H4.
      set (a5 := add_fact (Fact_accessor apply_node
                             (Field_call_witness Known_arity_code_pointer)
                             (node_of_simple (ap_callee ap))) a4).
      assert (S6 : steady a5
                     (match env_should_preserve_direct_calls env with
                      | Preserve_yes =>
                        add_apply env id (node_of_simple (ap_callee ap))
                          apply_node false a5
                      | Preserve_auto =>
                        add_apply env id (node_of_simple (ap_callee ap))
                          apply_node true a5
                      | Preserve_no =>
                        if negb (among id (env_unit_code_ids env))
                        then
                          add_apply env id (node_of_simple (ap_callee ap))
                            apply_node false a5
                        else a5
                      end)).
      { destruct (env_should_preserve_direct_calls env);
          try apply add_apply_steady.
        destruct (negb (among id (env_unit_code_ids env)));
          [ apply add_apply_steady | apply steady_refl ]. }
      set (a6 := match env_should_preserve_direct_calls env with
                 | Preserve_yes => _ | Preserve_auto => _ | Preserve_no => _
                 end) in S6 |- *.
      split.
      { refine (steady_trans _ _ _ (note_simple_steady _ _)
                  (steady_trans _ _ _ S4 _)).
        exact (steady_trans _ _ _ (add_facts_steady _ _) S6). }
      intros a' Hle.
      pose proof (acc_le_trans _ _ _ (proj1 S6) Hle) as Hle5.
      pose proof (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle5)
        as Hle4.
      destruct (H4 a' Hle4)
        as (witness & called_node & Hargs & Hrets & Hexnacc & Hcalled & Hused
            & Hcond & Hnoted).
      split; [ | split ].
      + exists witness, apply_node, called_node.
        do 6 (split; [ assumption | ]).
        apply facts_accessor. apply (add_facts_later _ _ _ _ Hle5). left.
        reflexivity.
      + apply (note_simple_noted _ a3 a').
        exact (acc_le_trans _ _ _ (proj1 S4) Hle4).
      + exact Hnoted.
    - pose proof (unknown_widget_spec env ap returns exn
                    (note_simple (ap_callee ap) a3)) as [S4 H4].
      destruct (make_unknown_arity_apply_widget env ap returns exn
                  (note_simple (ap_callee ap) a3)) as [apply_node a4] eqn:E4.
      simpl in S4, H4.
      split.
      { refine (steady_trans _ _ _ (note_simple_steady _ _)
                  (steady_trans _ _ _ S4 (add_facts_steady _ _))). }
      intros a' Hle.
      pose proof (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle)
        as Hle4.
      destruct (H4 a' Hle4)
        as (ws & called_node & Hlen & Hchain & Hused & Hcond & Hnoted).
      split; [ | split ].
      + exists ws, apply_node, called_node.
        do 4 (split; [ assumption | ]).
        apply facts_accessor. apply (add_facts_later _ _ _ _ Hle). left.
        reflexivity.
      + apply (note_simple_noted _ a3 a').
        exact (acc_le_trans _ _ _ (proj1 S4) Hle4).
      + exact Hnoted.
    - pose proof (known_widget_spec env ap returns exn
                    (note_simple (ap_callee ap) a3)) as [S4 H4].
      destruct (make_known_arity_apply_widget env ap returns exn
                  (note_simple (ap_callee ap) a3)) as [apply_node a4] eqn:E4.
      simpl in S4, H4.
      split.
      { refine (steady_trans _ _ _ (note_simple_steady _ _)
                  (steady_trans _ _ _ S4 (add_facts_steady _ _))). }
      intros a' Hle.
      pose proof (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle)
        as Hle4.
      destruct (H4 a' Hle4)
        as (witness & called_node & Hargs & Hrets & Hexnacc & Hcalled & Hused
            & Hcond & Hnoted).
      split; [ | split ].
      + exists witness, apply_node, called_node.
        do 6 (split; [ assumption | ]).
        apply facts_accessor. apply (add_facts_later _ _ _ _ Hle). left.
        reflexivity.
      + apply (note_simple_noted _ a3 a').
        exact (acc_le_trans _ _ _ (proj1 S4) Hle4).
      + exact Hnoted. }
  destruct Hkind as [S4 H4].
  split.
  { apply steady_adds.
    exact (steady_trans _ _ _ S1 (steady_trans _ _ _ S2
             (steady_trans _ _ _ S3 S4))). }
  intros a' Hle Hfrag. destruct (H4 a' Hle) as (Hcall & Hnc & Hna).
  split; [ exact Hfrag | split; [ exact Hnc | split; [ exact Hna | ] ] ].
  exists returns, exn. split; [ | split; [ | exact Hcall ] ].
  - intros params Hparams.
    rewrite (find_cont_bound _ _ _ _ _ Hparams) in E1.
    injection E1 as <- _. reflexivity.
  - intros p params Hparams.
    rewrite (find_cont_bound _ _ _ _ _ Hparams) in E2.
    injection E2 as <- _.
    destruct (Hexn p params eq_refl) as [-> Hle3]. split; [ reflexivity | ].
    exact (Hle3 a' (acc_le_trans _ _ _ (proj1 S4) Hle)).
Qed.

(* ------------------------------------------------------------------------ *)
(* Code                                                                     *)
(* ------------------------------------------------------------------------ *)

Lemma unknown_arity_witness_deps_steady (id : code_id) (returns : list node)
    (exn : node) (l : list (list node * node)) (a : acc) :
  steady a (unknown_arity_witness_deps id returns exn l a).
Proof.
  revert a. induction l as [ | [first witness] rest IH]; intros a;
    [ apply steady_refl | ].
  cbn [unknown_arity_witness_deps].
  refine (steady_trans _ _ _ (add_facts_steady _ _) _).
  destruct rest as [ | [p next] rest'];
    [ apply add_facts_steady | ].
  set (a1 := add_facts _ a).
  destruct (fresh a1) as [v a2] eqn:E.
  pose proof (steady_snd fresh a1 v a2 fresh_steady E) as S.
  exact (steady_trans _ _ _ S
           (steady_trans _ _ _ (add_facts_steady _ _) (IH _))).
Qed.

(* The witnesses of unknown arity of code. *)
Lemma code_chain_deps_spec (id : code_id) (returns : list node) (exn : node)
    (l : list (list node * node)) (a a' : acc) :
  acc_le (unknown_arity_witness_deps id returns exn l a) a' ->
  code_chain_edges (acc_graph a') id returns exn l.
Proof.
  revert a. induction l as [ | [first w] rest IH]; intros a Hle; [ exact I | ].
  set (l1 := map (fun '(i, p) => Fact_parameter w i p) (indexed first)
             ++ [Fact_constructor w Field_code_id_of_call_witness
                   (Node_code_id id)]).
  assert (H1 : ∀ a'', acc_le (add_facts l1 a) a'' ->
            (∀ (k : nat) (p : node),
               nth_error first k = Some p -> g_parameter (acc_graph a'') w k p)
            ∧ g_constructor (acc_graph a'') w Field_code_id_of_call_witness
                (Node_code_id id)).
  { intros a'' Hle''. split.
    - intros k p Hk. apply facts_parameter.
      apply (add_facts_later _ _ _ _ Hle''). apply in_or_app. left.
      apply in_map_iff. exists (k, p).
      split; [ reflexivity | exact (indexed_nth _ _ _ Hk) ].
    - apply facts_constructor. apply (add_facts_later _ _ _ _ Hle'').
      apply in_or_app. right. left. reflexivity. }
  destruct rest as [ | [first' w'] rest'].
  - cbn [unknown_arity_witness_deps] in Hle. cbn [code_chain_edges].
    destruct (H1 a' (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle))
      as [Hp Hcode].
    split; [ exact Hp | split; [ exact Hcode | split; [ | exact I ] ] ].
    split.
    + apply facts_constructor. apply (add_facts_later _ _ _ _ Hle). left.
      reflexivity.
    + intros i r Hr. apply facts_constructor.
      apply (add_facts_later _ _ _ _ Hle). right. apply in_map_iff.
      exists (i, r). split; [ reflexivity | exact (indexed_nth _ _ _ Hr) ].
  - cbn [unknown_arity_witness_deps] in Hle. cbn [code_chain_edges].
    fold l1 in Hle.
    destruct (fresh (add_facts l1 a)) as [v a2] eqn:E.
    pose proof (steady_snd fresh _ v a2 fresh_steady E) as S2.
    set (a3 := add_facts
                 [Fact_constructor w (Field_return_of_call (Normal 0)) v;
                  Fact_constructor v
                    (Field_call_witness Unknown_arity_code_pointer) w'] a2)
      in Hle.
    pose proof (acc_le_trans _ _ _
                  (proj1 (unknown_arity_witness_deps_steady id returns exn
                            ((first', w') :: rest') a3)) Hle) as Hle3.
    destruct (H1 a' (acc_le_trans _ _ _ (proj1 S2)
                       (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _))
                          Hle3)))
      as [Hp Hcode].
    split; [ exact Hp | split; [ exact Hcode | split ] ].
    + exists v. split; apply facts_constructor;
        apply (add_facts_later _ _ _ _ Hle3); [ left | right; left ];
        reflexivity.
    + exact (IH a3 Hle).
Qed.

Lemma unknown_witnesses_steady (id : code_id) (is_tupled : bool)
    (params_arity : arity) (params returns : list node) (exn : node)
    (a : acc) :
  steady a
    (snd (create_unknown_arity_call_witnesses id is_tupled params_arity
            params returns exn a)).
Proof.
  unfold create_unknown_arity_call_witnesses. destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses.
    destruct (fresh a) as [w a1] eqn:E1.
    pose proof (steady_snd fresh a w a1 fresh_steady E1) as S1.
    destruct (fresh a1) as [u a2] eqn:E2.
    pose proof (steady_snd fresh a1 u a2 fresh_steady E2) as S2.
    destruct (fresh a2) as [k a3] eqn:E3.
    pose proof (steady_snd fresh a2 k a3 fresh_steady E3) as S3.
    simpl. exact (steady_trans _ _ _ S1 (steady_trans _ _ _ S2
                    (steady_trans _ _ _ S3 (add_facts_steady _ _)))).
  - unfold create_unknown_arity_non_tupled_call_witnesses.
    match goal with
    | |- context [fresh_list ?n a] =>
      destruct (fresh_list n a) as [ws a1] eqn:E1;
      pose proof (steady_snd (fresh_list n) a ws a1 (fresh_list_steady n) E1)
        as S1
    end.
    simpl. exact (steady_trans _ _ _ S1
                    (unknown_arity_witness_deps_steady _ _ _ _ _)).
Qed.

(* [prepare_code] adds the code dep of the code, with the facts of its
   witness of known arity. *)
Lemma prepare_code_spec (id : code_id) (c : code) (a : acc) :
  ∃ cd : code_dep,
    acc_code_deps (prepare_code id c a) = (id, cd) :: acc_code_deps a
    ∧ acc_le a (prepare_code id c a)
    ∧ length (cd_params cd) = length (code_params_kinds c)
    ∧ length (cd_return cd) = length (code_result_kinds c)
    ∧ (∀ a' : acc,
         acc_le (prepare_code id c a) a' ->
         known_witness_edges (acc_graph a') id cd)
    ∧ (code_is_tupled c = false ->
       length (cd_unknown_arity_call_witnesses cd)
       = length (at_least_one_group
                   (group_by_parameter (code_params_arity c) (cd_params cd)))
       ∧ ∀ a' : acc,
           acc_le (prepare_code id c a) a' ->
           code_chain_edges (acc_graph a') id (cd_return cd) (cd_exn cd)
             (combine
                (at_least_one_group
                   (group_by_parameter (code_params_arity c) (cd_params cd)))
                (cd_unknown_arity_call_witnesses cd)))
    ∧ (code_is_tupled c = true ->
       ∀ a' : acc,
         acc_le (prepare_code id c a) a' ->
         tupled_witness_edges (acc_graph a') id cd).
Proof.
  unfold prepare_code.
  destruct (fresh_list (length (code_result_kinds c)) a) as [returns a1]
    eqn:E1.
  pose proof (steady_snd _ a returns a1 (fresh_list_steady _) E1) as S1.
  pose proof (fresh_list_length (length (code_result_kinds c)) a) as L1.
  rewrite E1 in L1.
  destruct (fresh a1) as [exn a2] eqn:E2.
  pose proof (steady_snd fresh a1 exn a2 fresh_steady E2) as S2.
  destruct (fresh a2) as [mc a3] eqn:E3.
  pose proof (steady_snd fresh a2 mc a3 fresh_steady E3) as S3.
  destruct (fresh_list (length (code_params_kinds c)) a3) as [params a4]
    eqn:E4.
  pose proof (steady_snd _ a3 params a4 (fresh_list_steady _) E4) as S4.
  pose proof (fresh_list_length (length (code_params_kinds c)) a3) as L4.
  rewrite E4 in L4.
  unfold create_known_arity_call_witness.
  destruct (fresh a4) as [w a5] eqn:E5.
  pose proof (steady_snd fresh a4 w a5 fresh_steady E5) as S5.
  set (facts_w :=
         map (fun '(i, p) => Fact_parameter w i p) (indexed params)
         ++ map (fun '(i, r) =>
                   Fact_constructor w (Field_return_of_call (Normal i)) r)
              (indexed returns)
         ++ [Fact_constructor w (Field_return_of_call Exn) exn;
             Fact_constructor w Field_code_id_of_call_witness
               (Node_code_id id)]).
  set (a6 := add_facts facts_w a5).
  pose proof (unknown_witnesses_steady id (code_is_tupled c)
                (code_params_arity c) params returns exn a6) as S7.
  destruct (create_unknown_arity_call_witnesses id (code_is_tupled c)
              (code_params_arity c) params returns exn a6)
    as [unknown a7] eqn:E7.
  simpl in S7.
  assert (Hunknown :
    code_is_tupled c = false ->
    length unknown
    = length (at_least_one_group
                (group_by_parameter (code_params_arity c) params))
    ∧ ∀ a' : acc,
        acc_le a7 a' ->
        code_chain_edges (acc_graph a') id returns exn
          (combine
             (at_least_one_group
                (group_by_parameter (code_params_arity c) params))
             unknown)).
  { intros Ht. unfold create_unknown_arity_call_witnesses in E7.
    rewrite Ht in E7.
    unfold create_unknown_arity_non_tupled_call_witnesses in E7.
    cbv zeta in E7.
    match type of E7 with
    | context [fresh_list ?n a6] =>
      pose proof (fresh_list_length n a6) as Lf;
      destruct (fresh_list n a6) as [ws a6'] eqn:Ef
    end.
    simpl in Lf. injection E7 as <- <-.
    split; [ exact Lf | intros a' Hle ].
    exact (code_chain_deps_spec _ _ _ _ _ _ Hle). }
  assert (Htupled :
    code_is_tupled c = true ->
    ∃ W U : node,
      unknown = [W]
      ∧ ∀ a' : acc,
          acc_le a7 a' ->
          (∀ (i : nat) (r : node),
             nth_error returns i = Some r ->
             g_constructor (acc_graph a') W (Field_return_of_call (Normal i))
               r)
          ∧ g_constructor (acc_graph a') W (Field_return_of_call Exn) exn
          ∧ g_constructor (acc_graph a') W Field_code_id_of_call_witness
              (Node_code_id id)
          ∧ g_parameter (acc_graph a') W 0 U
          ∧ ∀ (i : nat) (p : node),
              nth_error params i = Some p ->
              g_accessor (acc_graph a') p (Field_block i Value) U).
  { intros Ht. unfold create_unknown_arity_call_witnesses in E7.
    rewrite Ht in E7.
    unfold create_unknown_arity_tupled_call_witnesses in E7.
    destruct (fresh a6) as [W b1] eqn:F1.
    destruct (fresh b1) as [U b2] eqn:F2.
    destruct (fresh b2) as [Kp b3] eqn:F3.
    injection E7 as <- <-. exists W, U. split; [ reflexivity | ].
    intros a' Hle. split; [ | split; [ | split; [ | split ] ] ].
    - intros i r Hr. apply facts_constructor.
      apply (add_facts_later _ _ _ _ Hle). apply in_or_app. left.
      apply in_map_iff. exists (i, r).
      split; [ reflexivity | exact (indexed_nth _ _ _ Hr) ].
    - apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
      apply in_or_app. right. left. reflexivity.
    - apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
      apply in_or_app. right. right. left. reflexivity.
    - apply facts_parameter. apply (add_facts_later _ _ _ _ Hle).
      apply in_or_app. right. right. right. left. reflexivity.
    - intros i p Hp. apply facts_accessor.
      apply (add_facts_later _ _ _ _ Hle). apply in_or_app. right.
      right. right. right. apply in_or_app. left. apply in_map_iff.
      exists (i, p). split; [ reflexivity | exact (indexed_nth _ _ _ Hp) ]. }
  set (a8 := add_fact (Fact_any_source (Node_code_id id)) a7).
  assert (S : steady a a8).
  { refine (steady_trans _ _ _ S1 (steady_trans _ _ _ S2
              (steady_trans _ _ _ S3 (steady_trans _ _ _ S4
                 (steady_trans _ _ _ S5 (steady_trans _ _ _
                    (add_facts_steady _ _) (steady_trans _ _ _ S7
                       (add_facts_steady _ _)))))))). }
  eexists. simpl. split; [ rewrite <- (proj2 S); reflexivity | ].
  assert (Hle8 : acc_le a8
                   (add_code_dep id
                      {| cd_params := params; cd_my_closure := mc;
                         cd_return := returns; cd_exn := exn;
                         cd_known_arity_call_witness := w;
                         cd_unknown_arity_call_witnesses := unknown |} a8)).
  { repeat split; simpl; auto. }
  split; [ exact (acc_le_trans _ _ _ (proj1 S) Hle8) | ].
  split; [ exact L4 | split; [ exact L1 | ] ].
  assert (Hle7 : ∀ a' : acc,
            acc_le
              (add_code_dep id
                 {| cd_params := params; cd_my_closure := mc;
                    cd_return := returns; cd_exn := exn;
                    cd_known_arity_call_witness := w;
                    cd_unknown_arity_call_witnesses := unknown |} a8) a' ->
            acc_le a7 a').
  { intros a' Hle. refine (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _))
                             _).
    exact (acc_le_trans _ _ _ Hle8 Hle). }
  split; [ | split ].
  2: { intros Ht. destruct (Hunknown Ht) as [Lu Hu]. simpl.
       split; [ exact Lu | intros a' Hle; exact (Hu a' (Hle7 a' Hle)) ]. }
  2: { intros Ht. destruct (Htupled Ht) as (W & U & HWu & Hf).
       intros a' Hle. simpl. exists W, U. split; [ exact HWu | ].
       exact (Hf a' (Hle7 a' Hle)). }
  intros a' Hle. pose proof (acc_le_trans _ _ _ Hle8 Hle) as Hle'.
  assert (Hle6 : acc_le a6 a').
  { refine (acc_le_trans _ _ _ (proj1 S7) _).
    exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _)) Hle'). }
  simpl. split; [ | split; [ | split; [ | split ] ] ].
  - intros i p Hnth. apply facts_parameter.
    apply (add_facts_later _ _ _ _ Hle6). apply in_or_app. left.
    apply in_map_iff. exists (i, p).
    split; [ reflexivity | exact (indexed_nth _ _ _ Hnth) ].
  - intros i r Hnth. apply facts_constructor.
    apply (add_facts_later _ _ _ _ Hle6). apply in_or_app. right.
    apply in_or_app. left. apply in_map_iff. exists (i, r).
    split; [ reflexivity | exact (indexed_nth _ _ _ Hnth) ].
  - apply facts_constructor. apply (add_facts_later _ _ _ _ Hle6).
    apply in_or_app. right. apply in_or_app. right. left. reflexivity.
  - apply facts_constructor. apply (add_facts_later _ _ _ _ Hle6).
    apply in_or_app. right. apply in_or_app. right. right. left.
    reflexivity.
  - apply facts_any_source. apply (add_facts_later _ _ _ _ Hle'). left.
    reflexivity.
Qed.

(* What the code dep [cd] of the code [c] has, with the facts of its
   witnesses in the graph of [a']. *)
Definition code_dep_made (id : code_id) (c : code) (cd : code_dep) (a' : acc)
    : Prop :=
  known_witness_edges (acc_graph a') id cd
  ∧ (code_is_tupled c = false ->
     code_chain_edges (acc_graph a') id (cd_return cd) (cd_exn cd)
       (combine
          (at_least_one_group
             (group_by_parameter (code_params_arity c) (cd_params cd)))
          (cd_unknown_arity_call_witnesses cd)))
  ∧ (code_is_tupled c = true -> tupled_witness_edges (acc_graph a') id cd).

Definition code_dep_lengths (c : code) (cd : code_dep) : Prop :=
  length (cd_params cd) = length (code_params_kinds c)
  ∧ length (cd_return cd) = length (code_result_kinds c)
  ∧ (code_is_tupled c = false ->
     length (cd_unknown_arity_call_witnesses cd)
     = length (at_least_one_group
                 (group_by_parameter (code_params_arity c) (cd_params cd)))).

(* The code deps that a group prepares, the last first. *)
Lemma prepare_bindings_spec (bindings : list static_binding) (a : acc) :
  acc_le a (List.fold_left (fun a b => prepare_binding b a) bindings a)
  ∧ ∃ l : finite_map code_dep,
      acc_code_deps (List.fold_left (fun a b => prepare_binding b a) bindings a)
      = l ++ acc_code_deps a
      ∧ map fst l = rev (flat_map code_ids_of_binding bindings)
      ∧ ∀ (id : code_id) (c : code),
          Bind_code id c ∈ bindings ->
          ∃ cd : code_dep,
            (id, cd) ∈ l
            ∧ code_dep_lengths c cd
            ∧ ∀ a' : acc,
                acc_le
                  (List.fold_left (fun a b => prepare_binding b a) bindings a)
                  a' ->
                code_dep_made id c cd a'.
Proof.
  revert a. induction bindings as [ | b bindings IH]; intros a; simpl.
  - split; [ apply acc_le_refl | exists []; split; [ reflexivity | ] ].
    split; [ reflexivity | intros ? ? [] ].
  - assert (Hb : ∃ lb : finite_map code_dep,
              acc_code_deps (prepare_binding b a) = lb ++ acc_code_deps a
              ∧ map fst lb = code_ids_of_binding b
              ∧ acc_le a (prepare_binding b a)
              ∧ ∀ (id : code_id) (c : code),
                  b = Bind_code id c ->
                  ∃ cd : code_dep,
                    (id, cd) ∈ lb
                    ∧ code_dep_lengths c cd
                    ∧ ∀ a' : acc,
                        acc_le (prepare_binding b a) a' ->
                        code_dep_made id c cd a').
    { destruct b as [id c | id | s const]; simpl.
      - destruct (prepare_code_spec id c a)
          as (cd & Hcds & Hle & L1 & L2 & Hw & Hu & Htu).
        exists [(id, cd)]. split; [ exact Hcds | split; [ reflexivity | ] ].
        split; [ exact Hle | ].
        intros id' c' [= <- <-]. exists cd. split; [ left; reflexivity | ].
        split.
        + split; [ exact L1 | split; [ exact L2 | ] ].
          intros Ht. exact (proj1 (Hu Ht)).
        + intros a' Hle'. split; [ exact (Hw a' Hle') | split ].
          * intros Ht. exact (proj2 (Hu Ht) a' Hle').
          * intros Ht. exact (Htu Ht a' Hle').
      - exists []. split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ apply acc_le_refl | intros ? ? ? ; discriminate ].
      - exists []. split; [ reflexivity | split; [ reflexivity | ] ].
        split; [ apply acc_le_refl | intros ? ? ? ; discriminate ]. }
    destruct Hb as (lb & Hcds & Hmap & Hle & Hcode).
    destruct (IH (prepare_binding b a)) as [Hle' (l & Hl & Hmap' & Hcodes)].
    split; [ exact (acc_le_trans _ _ _ Hle Hle') | ].
    exists (l ++ lb). split; [ rewrite Hl, Hcds, app_assoc; reflexivity | ].
    split; [ rewrite map_app, Hmap', Hmap, rev_app_distr | ].
    + f_equal. destruct b; reflexivity.
    + intros id c [-> | Hin].
      * destruct (Hcode id c eq_refl) as (cd & Hin & L & Hw).
        exists cd. split; [ apply in_or_app; right; exact Hin | ].
        split; [ exact L | ].
        intros a' Hle''. apply Hw. exact (acc_le_trans _ _ _ Hle' Hle'').
      * destruct (Hcodes id c Hin) as (cd & Hin' & L & Hw).
        exists cd. split; [ apply in_or_app; left; exact Hin' | ].
        auto.
Qed.

(* A binding of a list whose names are distinct is the one that
   [lookup] finds. *)
Lemma lookup_app_nodup {A : Type} (l r : finite_map A) (k : positive)
    (v : A) :
  NoDup (map fst l) -> (k, v) ∈ l -> lookup k (l ++ r) = Some v.
Proof.
  induction l as [ | [k' v'] l IH]; intros Hnd Hin; [ destruct Hin | ].
  simpl in Hnd. inversion Hnd as [ | ? ? Hnin Hnd']; subst.
  simpl. destruct Hin as [[= -> ->] | Hin].
  - rewrite Pos.eqb_refl. reflexivity.
  - destruct (Pos.eqb_spec k k') as [-> | _]; [ | exact (IH Hnd' Hin) ].
    exfalso. apply Hnin. apply (in_map fst) in Hin. exact Hin.
Qed.

(* Code that binds nothing static has no code id among its static
   names. *)
Lemma binds_statics_no_code_ids :
  (∀ e : expr,
     binds_statics e = false -> static_names code_ids_of_binding e = [])
  ∧ (∀ h : continuation_handler,
       handler_binds_statics h = false ->
       handler_static_names code_ids_of_binding h = []).
Proof.
  apply expr_handler_ind; simpl; auto.
  - intros bindings body _ IH H. apply orb_false_iff in H as [Hb H].
    rewrite (IH H), app_nil_r.
    induction bindings as [ | [ | | ] bindings IHb]; simpl in Hb |- *;
      try discriminate; auto.
  - intros k h body IHh IH H. apply orb_false_iff in H as [Hh H].
    rewrite (IHh Hh), (IH H). reflexivity.
  - intros handlers body IHh IH H. apply orb_false_iff in H as [Hh H].
    rewrite (IH H), app_nil_r.
    induction handlers as [ | [k h] handlers IHl]; [ reflexivity | ].
    simpl in Hh |- *. apply orb_false_iff in Hh as [Hh1 Hh2].
    rewrite (IHh k h (or_introl eq_refl) Hh1).
    apply IHl; [ | exact Hh2 ].
    intros k' h' Hin. exact (IHh k' h' (or_intror Hin)).
Qed.

(* ------------------------------------------------------------------------ *)
(* The traversal                                                            *)
(* ------------------------------------------------------------------------ *)

Lemma distinct_app_l (l r : list positive) :
  distinct (l ++ r) = true -> distinct l = true.
Proof.
  rewrite !distinct_NoDup. apply NoDup_app_remove_r.
Qed.

Lemma distinct_app_r (l r : list positive) :
  distinct (l ++ r) = true -> distinct r = true.
Proof.
  rewrite !distinct_NoDup. apply NoDup_app_remove_l.
Qed.

Lemma distinct_flat_map {A : Type} (f : A -> list positive) (l : list A)
    (x : A) :
  distinct (flat_map f l) = true -> x ∈ l -> distinct (f x) = true.
Proof.
  induction l as [ | y l IH]; intros Hd Hin; [ destruct Hin | ].
  simpl in Hd. destruct Hin as [<- | Hin].
  - exact (distinct_app_l _ _ Hd).
  - exact (IH (distinct_app_r _ _ Hd) Hin).
Qed.

(* A fold whose steps add no code dep, from given code deps. *)
Lemma fold_left_steady_from {A : Type} (f : acc -> A -> acc)
    (P : A -> acc -> Prop) (l : list A) (cds0 : finite_map code_dep) :
  (∀ x : A, x ∈ l ->
     ∀ a : acc, acc_code_deps a = cds0 ->
       steady a (f a x) ∧ ∀ a', acc_le (f a x) a' -> P x a') ->
  ∀ a : acc, acc_code_deps a = cds0 ->
    steady a (List.fold_left f l a)
    ∧ ∀ a', acc_le (List.fold_left f l a) a' -> ∀ x : A, x ∈ l -> P x a'.
Proof.
  induction l as [ | x l IH]; intros H a Hcds; simpl.
  - split; [ apply steady_refl | intros a' _ y [] ].
  - destruct (H x (or_introl eq_refl) a Hcds) as [Hs HP].
    destruct (IH (fun y Hy => H y (or_intror Hy)) (f a x)
                (eq_trans (proj2 Hs) Hcds)) as [Hs' HP'].
    split; [ exact (steady_trans _ _ _ Hs Hs') | ].
    intros a' Hle y [<- | Hy].
    + apply HP. exact (acc_le_trans _ _ _ (proj1 Hs') Hle).
    + exact (HP' a' Hle y Hy).
Qed.

Lemma adds_perm (ids ids' : list code_id) (a a' : acc) :
  adds ids a a' -> Permutation ids ids' -> adds ids' a a'.
Proof.
  intros [Hle (l & Hl & Hp)] Hperm.
  split; [ exact Hle | exists l; split;
    [ exact Hl | exact (Permutation_trans Hp Hperm) ] ].
Qed.

Lemma code_body_projected (bindings : list static_binding) (id : code_id)
    (c : code) (w : value_slot) :
  Bind_code id c ∈ bindings ->
  w ∈ projected_value_slots (code_body c) ->
  w ∈ flat_map binding_projected_value_slots bindings.
Proof.
  intros Hin Hw. apply in_flat_map. exists (Bind_code id c).
  split; [ exact Hin | destruct c; exact Hw ].
Qed.

Lemma slots_and_l (sr : Prop) (b b' : bool) :
  sr ∨ (b && b') = true -> sr ∨ b = true.
Proof.
  intros [H | H]; [ left; exact H | right ].
  apply andb_true_iff in H. exact (proj1 H).
Qed.

Lemma slots_and_r (sr : Prop) (b b' : bool) :
  sr ∨ (b && b') = true -> sr ∨ b' = true.
Proof.
  intros [H | H]; [ left; exact H | right ].
  apply andb_true_iff in H. exact (proj2 H).
Qed.

Lemma slots_forallb {A : Type} (sr : Prop) (f : A -> bool) (l : list A)
    (x : A) :
  sr ∨ forallb f l = true -> x ∈ l -> sr ∨ f x = true.
Proof.
  intros [H | H] Hin; [ left; exact H | right ].
  rewrite forallb_forall in H. exact (H x Hin).
Qed.

(* Code that binds nothing static has no code of the unit. *)
Lemma unit_codes_of_no_statics :
  (∀ e : expr, binds_statics e = false -> unit_codes e = [])
  ∧ (∀ h : continuation_handler,
       handler_binds_statics h = false -> handler_unit_codes h = []).
Proof.
  apply expr_handler_ind; simpl; auto.
  - intros bindings body _ IH H. apply orb_false_iff in H as [Hb H].
    rewrite (IH H), app_nil_r.
    induction bindings as [ | [ | | ] bindings IHb]; simpl in Hb |- *;
      try discriminate; auto.
  - intros k h body IHh IH H. apply orb_false_iff in H as [Hh H].
    rewrite (IHh Hh), (IH H). reflexivity.
  - intros handlers body IHh IH H. apply orb_false_iff in H as [Hh H].
    rewrite (IH H), app_nil_r.
    induction handlers as [ | [k h] handlers IHl]; [ reflexivity | ].
    simpl in Hh |- *. apply orb_false_iff in Hh as [Hh1 Hh2].
    rewrite (IHh k h (or_introl eq_refl) Hh1).
    apply IHl; [ | exact Hh2 ].
    intros k' h' Hin. exact (IHh k' h' (or_intror Hin)).
Qed.

Section Traverse.

Variable codes : list (code_id * code).
Variable symbols : list symbol.
Variable K : list value_slot.
Variable slots_read : Prop.
(* Whether the primitives of the heap are in the fragment. *)
Variable hb : bool.

Abbreviation keep := (fun w : value_slot => among w K).
Abbreviation heap_prims := (if hb then True else False).

Definition edges_in (ctx : option code_id) (ce : finite_map (list node))
    (e : expr) (a' : acc) : Prop :=
  expr_edges (acc_graph a') (acc_noted a') (acc_deps a') (acc_code_deps a')
    keep codes symbols slots_read heap_prims ctx ce e.

(* What a static binding makes before code is traversed. *)
Definition static_binding_made (binding : static_binding) (a' : acc) : Prop :=
  match binding with
  | Bind_symbol s (Static_const_block _ fields) =>
    block_edges (acc_graph a') (acc_noted a') (Node_symbol s) fields
  | Bind_symbol s (Static_const_immutable_value_array []) =>
    g_alias (acc_graph a') (Node_symbol s) Node_all_constants
  | Bind_symbol s (Static_const_immutable_value_array elements) =>
    block_edges (acc_graph a') (acc_noted a') (Node_symbol s) elements
  | Bind_symbol s (Static_const_boxed_number contents) =>
    boxed_edges (acc_graph a') s contents
  | Bind_symbol s (Static_const_immutable_string _) =>
    g_alias (acc_graph a') (Node_symbol s) Node_all_constants
  | Bind_symbol s
      (Static_const_closure_of_set slot closure_symbols function_decls
         value_slots) =>
    static_set_edges (acc_graph a') (acc_noted a') (acc_deps a') keep symbols
      s slot closure_symbols function_decls value_slots
  | _ => True
  end.

Lemma should_keep (env : traverse_env) :
  env_value_slots_to_keep env = K ->
  ∀ w : value_slot, should_keep_value_slot env w = keep w.
Proof. intros HK w. unfold should_keep_value_slot. rewrite HK. reflexivity.
Qed.

(* The facts of a term of the fragment, and of a handler. *)
Lemma traverse_makes :
  (∀ e : expr,
     flow_fragment hb codes symbols e = true ->
     distinct (static_names code_ids_of_binding e) = true ->
     (∀ w : value_slot, w ∈ projected_value_slots e -> keep w = true) ->
     slots_read ∨ projection_free e = true ->
     (∀ x : code_id * code, x ∈ unit_codes e -> x ∈ codes) ->
     ∀ env : traverse_env,
       env_value_slots_to_keep env = K ->
       makes (static_names code_ids_of_binding e) (traverse env e)
         (edges_in (env_current_code_id env) (env_conts env) e))
  ∧ (∀ h : continuation_handler,
       handler_flow_fragment hb codes symbols h = true ->
       distinct (handler_static_names code_ids_of_binding h) = true ->
       (∀ w : value_slot,
          w ∈ handler_projected_value_slots h -> keep w = true) ->
       slots_read ∨ handler_projection_free h = true ->
       (∀ x : code_id * code, x ∈ handler_unit_codes h -> x ∈ codes) ->
       ∀ env : traverse_env,
         env_value_slots_to_keep env = K ->
         makes (handler_static_names code_ids_of_binding h)
           (traverse_handler env h)
           (fun a' =>
              handler_edges (acc_graph a') (acc_noted a') (acc_deps a')
                (acc_code_deps a') keep codes symbols slots_read heap_prims
                (env_current_code_id env) (env_conts env) h)).
Proof.
  apply expr_handler_ind.
  - (* [Let] *)
    intros x kind d body IH Hfrag Hdist Hproj Hslots Hcodes_in env HK.
    simpl in *.
    apply andb_true_iff in Hfrag as [Hd Hbody].
    assert (Hsd : named_projection_free d = false -> slots_read).
    { intros Hnp. destruct (slots_and_l _ _ _ Hslots) as [H | H];
        [ exact H | congruence ]. }
    pose proof (traverse_named_makes env slots_read hb x d Hd Hsd) as Hn.
    refine (makes_compose _ _ _ _ _ _ (makes_weaken _ _ _ _ _ (Hn _))
              (IH Hbody Hdist _ (slots_and_r _ _ _ Hslots) Hcodes_in env HK)).
    + intros a' H. unfold named_edges in H |- *.
      destruct d; simpl in H |- *; try exact H.
      * intros id0 Hid0. destruct (H id0 Hid0) as (sites & Hx & Hall).
        exists sites. split; [ exact Hx | ].
        intros f id Hf. destruct (Hall f id Hf) as (Hdep & Hfs & Hvs).
        split; [ exact Hdep | split; [ exact Hfs | ] ].
        intros w s Hl Hk. apply Hvs; [ exact Hl | ].
        rewrite (should_keep env HK). exact Hk.
      * destruct H as (H & Hk & Hs). split; [ exact H | split; [ | exact Hs ] ].
        rewrite <- (should_keep env HK). exact Hk.
    + intros w Hw. rewrite (should_keep env HK).
      apply Hproj. apply in_or_app. left. exact Hw.
    + intros w Hw. apply Hproj. apply in_or_app. right. exact Hw.
  - (* [Let_static] *)
    intros bindings body IHcode IH Hfrag Hdist Hproj Hslots Hcodes_in env HK.
    simpl in *.
    apply andb_true_iff in Hfrag as [Hbindings Hbody].
    rewrite forallb_forall in Hbindings.
    pose proof (distinct_app_l _ _ Hdist) as Hdist_b.
    pose proof (distinct_app_r _ _ Hdist) as Hdist_body.
    intros a.
    change (traverse env (Let_static bindings body) a)
      with (traverse env body
              (List.fold_left (fun a b => traverse_binding env b a) bindings
                 (List.fold_left (fun a b => traverse_static_binding env b a)
                    bindings
                    (List.fold_left (fun a b => prepare_binding b a)
                       bindings a)))).
    destruct (prepare_bindings_spec bindings a)
      as [Hle_p (l & Hl & Hmap & Hcodes)].
    set (a_p := List.fold_left (fun a b => prepare_binding b a) bindings a)
      in *.
    assert (Hnd : NoDup (map fst l)).
    { rewrite Hmap. apply NoDup_rev. apply distinct_NoDup. exact Hdist_b. }
    (* The static constants. *)
    assert (Hstatic :
      makes (flat_map (fun _ => []) bindings)
        (fun a => List.fold_left (fun a b => traverse_static_binding env b a)
                    bindings a)
        (fun a' => ∀ b, b ∈ bindings -> static_binding_made b a')).
    { apply fold_left_makes. intros b Hin.
      specialize (Hbindings b Hin).
      destruct b as [id c | id | s [ | slot syms decls vs | t fields
                                   | contents | bytes | elements ]];
        simpl in Hbindings; try discriminate Hbindings;
        try (apply makes_steady; [ intros; apply steady_refl
                                 | intros; exact I ]).
      - eapply makes_weaken; [ | exact (static_set_makes env symbols s slot
                                          syms decls vs Hbindings) ].
        intros a' H. simpl. destruct H as (H0 & H1 & H2 & H3).
        split; [ exact H0 | split; [ exact H1 | split; [ exact H2 | ] ] ].
        intros w x Hl' Hk. apply H3; [ exact Hl' | ].
        rewrite (should_keep env HK). exact Hk.
      - exact (block_deps_makes (Node_symbol s) fields).
      - exact (boxed_number_makes s contents).
      - exact (alias_constants_makes (Node_symbol s)).
      - destruct elements as [ | element elements ];
          [ exact (alias_constants_makes (Node_symbol s))
          | exact (block_deps_makes (Node_symbol s) (element :: elements)) ].
    }
    destruct (Hstatic a_p) as [[Hle_s (ls & Hls & Hperm_s)] HPs].
    assert (Hls_nil : ls = []).
    { assert (Hnil : flat_map (fun _ : static_binding => @nil code_id)
                       bindings = [])
        by (clear; induction bindings; auto).
      rewrite Hnil in Hperm_s. apply Permutation_sym, Permutation_nil
        in Hperm_s.
      destruct ls; [ reflexivity | discriminate ]. }
    subst ls. simpl in Hls.
    set (a_s := List.fold_left (fun a b => traverse_static_binding env b a)
                  bindings a_p) in *.
    (* The code. *)
    set (code_made := fun (b : static_binding) (a' : acc) =>
      match b with
      | Bind_code id c =>
        ∃ cd : code_dep,
          lookup id (acc_code_deps a_p) = Some cd
          ∧ g_alias (acc_graph a') (Node_variable (code_my_closure c))
              (cd_my_closure cd)
          ∧ (∀ (p : variable) (q : node),
               (p, q) ∈ combine (code_params c) (cd_params cd) ->
               g_alias (acc_graph a') (Node_variable p) q)
          ∧ edges_in (Some id) (code_conts c cd) (code_body c) a'
      | _ => True
      end).
    assert (Hcode :
      ∀ b : static_binding, b ∈ bindings ->
        ∀ a0 : acc, acc_code_deps a0 = acc_code_deps a_p ->
          steady a0 (traverse_binding env b a0)
          ∧ ∀ a', acc_le (traverse_binding env b a0) a' -> code_made b a').
    { intros b Hin a0 Hcds0. specialize (Hbindings b Hin).
      destruct b as [id c | id | s const];
        try (split; [ apply steady_refl | intros; exact I ]).
      destruct c as [params mc ret exn metadata cbody].
      simpl in Hbindings.
      apply andb_true_iff in Hbindings as [Hb Hcb].
      apply andb_true_iff in Hb as [Hb Hcounted].
      apply andb_true_iff in Hb as [Hne Hbinds].
      destruct (Hcodes id (Code params mc ret exn metadata cbody) Hin)
        as (cd & Hcd & _).
      assert (Hlookup : lookup id (acc_code_deps a0) = Some cd).
      { rewrite Hcds0, Hl. exact (lookup_app_nodup _ _ _ _ Hnd Hcd). }
      simpl. rewrite Hlookup.
      set (a1 := add_facts _ a0).
      assert (Hnames : static_names code_ids_of_binding cbody = []).
      { apply (proj1 binds_statics_no_code_ids).
        apply negb_true_iff in Hbinds. exact Hbinds. }
      destruct (IHcode id (Code params mc ret exn metadata cbody) Hin Hcb
                  ltac:(simpl; rewrite Hnames; reflexivity)
                  (fun w Hw => Hproj w (in_or_app _ _ _ (or_introl
                     (code_body_projected bindings id
                        (Code params mc ret exn metadata cbody) w Hin Hw))))
                  (slots_forallb _ _ _ _ (slots_and_l _ _ _ Hslots) Hin)
                  ltac:(intros x Hx; simpl in Hx;
                        rewrite (proj1 unit_codes_of_no_statics _
                                   (proj1 (negb_true_iff _) Hbinds)) in Hx;
                        destruct Hx)
                  (code_env env id ret exn cd) HK a1)
        as [[Hle1 (lb & Hlb & Hpermb)] HPb].
      cbn [code_body] in Hpermb.
      rewrite Hnames in Hpermb. apply Permutation_sym, Permutation_nil
        in Hpermb.
      destruct lb; [ | discriminate ]. simpl in Hlb.
      split.
      + split; [ exact (acc_le_trans _ _ _ (proj1 (add_facts_steady _ _))
                          Hle1) | exact Hlb ].
      + intros a' Hle'. exists cd. split; [ rewrite <- Hcds0; exact Hlookup | ].
        pose proof (acc_le_trans _ _ _ Hle1 Hle') as Hle''.
        split; [ | split ].
        * apply facts_alias. apply (add_facts_later _ _ _ _ Hle''). left.
          reflexivity.
        * intros p q Hpq. apply facts_alias.
          apply (add_facts_later _ _ _ _ Hle''). right.
          apply in_map_iff. exists (p, q). auto.
        * exact (HPb a' Hle'). }
    destruct (fold_left_steady_from _ code_made bindings (acc_code_deps a_p)
                Hcode a_s Hls) as [Hs_c HPc].
    set (a_c := List.fold_left (fun a b => traverse_binding env b a)
                  bindings a_s) in *.
    destruct (IH Hbody Hdist_body
                (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
                (slots_and_r _ _ _ Hslots)
                (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
                env HK a_c) as [Hadds_b HPb].
    split.
    + refine (adds_perm _ _ _ _ (adds_trans _ _ _ _ _
                (adds_trans _ _ _ _ _ (conj Hle_p (ex_intro _ l (conj Hl
                   (eq_ind_r (fun m => Permutation m _) (Permutation_refl _)
                      Hmap))))
                   (steady_adds _ _ (steady_trans _ _ _
                      (conj Hle_s Hls) Hs_c))) Hadds_b) _).
      rewrite app_nil_r. apply Permutation_app_tail.
      apply Permutation_sym, Permutation_rev.
    + intros a' Hle'. unfold edges_in. simpl. split; [ | exact (HPb a' Hle') ].
      pose proof (acc_le_trans _ _ _ (proj1 Hadds_b) Hle') as Hle_c.
      pose proof (acc_le_trans _ _ _ (proj1 Hs_c) Hle_c) as Hle_sc.
      apply (every_in _ _ _). intros b Hin.
      specialize (Hbindings b Hin).
      destruct b as [id c | id | s [ | slot syms decls vs | t fields | | | ]];
        simpl in Hbindings; try discriminate Hbindings; simpl.
      * destruct c as [params mc ret exn metadata cbody].
        simpl in Hbindings.
        apply andb_true_iff in Hbindings as [Hb Hcb].
        apply andb_true_iff in Hb as [Hb Hcounted].
        apply andb_true_iff in Hb as [Hne Hbinds].
        destruct (Hcodes id (Code params mc ret exn metadata cbody) Hin)
          as (cd & Hcd & (L1 & L2 & Lu) & Hw).
        destruct (HPc a' Hle_c _ Hin) as (cd' & Hlookup & Hmc & Hps & Hbody').
        rewrite Hl, (lookup_app_nodup _ _ _ _ Hnd Hcd) in Hlookup.
        injection Hlookup as <-.
        split.
        { apply Hcodes_in. apply in_or_app. left. apply in_flat_map.
          exists (Bind_code id (Code params mc ret exn metadata cbody)).
          split; [ exact Hin | left; reflexivity ]. }
        exists cd. split; [ | split; [ | split; [ | split; [ | split;
          [ | split; [ | split; [ | split; [ | split; [ | split; [ | split ]
          ] ] ] ] ] ] ] ] ].
        -- apply (proj2 (proj2 (proj2 Hle_c))).
           apply (proj2 (proj2 (proj2 (proj1 Hs_c)))).
           apply (proj2 (proj2 (proj2 Hle_s))). rewrite Hl.
           apply in_or_app. left. exact Hcd.
        -- exact (proj1 (Hw a' (acc_le_trans _ _ _ Hle_s Hle_sc))).
        -- intros Ht. split; [ exact (Lu Ht) | ].
           exact (proj1 (proj2 (Hw a' (acc_le_trans _ _ _ Hle_s Hle_sc))) Ht).
        -- intros Ht.
           exact (proj2 (proj2 (Hw a' (acc_le_trans _ _ _ Hle_s Hle_sc))) Ht).
        -- exact L1.
        -- exact L2.
        -- exact Hmc.
        -- exact Hps.
        -- apply Pos.eqb_neq. apply negb_true_iff. exact Hne.
        -- apply negb_true_iff. exact Hbinds.
        -- exact Hcounted.
        -- exact Hbody'.
      * exact I.
      * exact (HPs a' Hle_sc _ Hin).
      * exact (HPs a' Hle_sc _ Hin).
      * exact (HPs a' Hle_sc _ Hin).
      * exact (HPs a' Hle_sc _ Hin).
      * exact (HPs a' Hle_sc _ Hin).
  - (* [Let_cont_non_recursive] *)
    intros k h body IHh IH Hfrag Hdist Hproj Hslots Hcodes_in env HK.
    simpl in *.
    apply andb_true_iff in Hfrag as [Hh Hbody].
    pose proof (makes_compose _ _ _ _ _ _
                  (makes_compose _ _ _ _ _ _
                     (IHh Hh (distinct_app_l _ _ Hdist)
                        (fun w Hw => Hproj w (in_or_app _ _ _ (or_introl Hw)))
                        (slots_and_l _ _ _ Hslots)
                        (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_introl
                          Hx)))
                        env HK)
                     (makes_steady _ (fun _ => True)
                        (exn_handler_deps_steady env h) (fun _ _ _ => I)))
                  (IH Hbody (distinct_app_r _ _ Hdist)
                     (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
                     (slots_and_r _ _ _ Hslots)
                     (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
                     (add_cont env k (handler_params h)) HK)) as Hboth.
    rewrite app_nil_r in Hboth.
    eapply makes_weaken; [ | exact Hboth ].
    intros a' [[Hh' _] Hb]. split; [ exact Hh' | exact Hb ].
  - (* [Let_cont_recursive] *)
    intros handlers body IHh IH Hfrag Hdist Hproj Hslots Hcodes_in env HK.
    simpl in *.
    apply andb_true_iff in Hfrag as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    set (env' := add_rec_conts env handlers).
    destruct (add_rec_conts_fields env handlers) as [Hctx HK'].
    apply (makes_ext _
             (fun a =>
                traverse env' body
                  (List.fold_left (fun a '(_, h) => traverse_handler env' h a)
                     handlers a))); [ reflexivity | ].
    assert (Hfold :
      makes (flat_map (fun '(_, h) => handler_static_names
                                        code_ids_of_binding h) handlers)
        (fun a =>
           List.fold_left (fun a '(_, h) => traverse_handler env' h a)
             handlers a)
        (fun a' =>
           ∀ kh, kh ∈ handlers ->
             (fun '(_, h) =>
                handler_edges (acc_graph a') (acc_noted a') (acc_deps a')
                  (acc_code_deps a') keep codes symbols slots_read heap_prims
                  (env_current_code_id env') (env_conts env') h) kh)).
    { apply fold_left_makes. intros [k h] Hin.
      apply (IHh k h Hin (Hhandlers _ Hin)).
      - exact (distinct_flat_map _ _ _ (distinct_app_l _ _ Hdist) Hin).
      - intros w Hw. apply Hproj. apply in_or_app. left.
        apply in_flat_map. exists (k, h). auto.
      - exact (slots_forallb _ _ _ _ (slots_and_l _ _ _ Hslots) Hin).
      - intros x Hx. apply Hcodes_in. apply in_or_app. left.
        apply in_flat_map. exists (k, h). auto.
      - unfold env'. rewrite HK'. exact HK. }
    pose proof (makes_compose _ _ _ _ _ _ Hfold
                  (IH Hbody (distinct_app_r _ _ Hdist)
                     (fun w Hw => Hproj w (in_or_app _ _ _ (or_intror Hw)))
                     (slots_and_r _ _ _ Hslots)
                     (fun x Hx => Hcodes_in x (in_or_app _ _ _ (or_intror Hx)))
                     env' ltac:(unfold env'; rewrite HK'; exact HK)))
      as Hboth.
    eapply makes_weaken; [ | exact Hboth ].
    intros a' [Hall Hb]. unfold edges_in in Hb |- *. simpl.
    unfold env' in Hall, Hb.
    rewrite add_rec_conts_conts, Hctx in Hall, Hb. split; [ | exact Hb ].
    apply (every_in _ _ _). exact Hall.
  - (* [Apply] *)
    intros ap Hfrag _ _ _ _ env _. simpl in Hfrag.
    eapply makes_weaken; [ | exact (traverse_apply_makes env codes ap) ].
    intros a' H. exact (H Hfrag).
  - (* [Apply_cont] *)
    intros ac _ _ _ _ _ env _. apply apply_cont_deps_makes.
  - (* [Switch] *)
    intros scrutinee arms _ _ _ _ _ env _. apply traverse_switch_makes.
  - (* [Invalid] *)
    intros _ _ _ _ _ env _ a.
    split; [ exact (steady_adds _ _ (steady_refl a)) | ].
    intros; exact I.
  - (* A handler *)
    intros params metadata handler IH Hfrag Hdist Hproj Hslots Hcodes_in env HK.
    exact (IH Hfrag Hdist Hproj Hslots Hcodes_in env HK).
Qed.

End Traverse.

(* ------------------------------------------------------------------------ *)
(* The delayed deps and the unit                                            *)
(* ------------------------------------------------------------------------ *)

(* What the closure dep of a closure gives it: with the code of the unit,
   its call witnesses of known arity and of unknown arity (the first of
   the code) and [my_closure], used if the code is; with other code, a
   witness that has any source. *)
Definition closure_dep_facts (G : graph) (cds : finite_map code_dep)
    (d : closure_dep) : Prop :=
  match lookup (cld_closure_code_id d) cds with
  | Some cd =>
    g_propagate G (Node_code_id (cld_closure_code_id d)) (cd_my_closure cd)
      (cld_let_bound_name_of_the_closure d)
    ∧ g_constructor G (cld_let_bound_name_of_the_closure d)
        (Field_call_witness Known_arity_code_pointer)
        (cd_known_arity_call_witness cd)
    ∧ ∀ w : node,
        hd_error (cd_unknown_arity_call_witnesses cd) = Some w ->
        g_constructor G (cld_let_bound_name_of_the_closure d)
          (Field_call_witness Unknown_arity_code_pointer) w
  | None =>
    ∃ witness : node,
      g_any_source G witness
      ∧ g_constructor G (cld_let_bound_name_of_the_closure d)
          (Field_call_witness Known_arity_code_pointer) witness
  end.

Lemma add_closure_dep_spec (cds : finite_map code_dep) (d : closure_dep)
    (a : acc) :
  steady a (add_closure_dep cds d a)
  ∧ acc_closure_deps (add_closure_dep cds d a) = acc_closure_deps a
  ∧ acc_symbols (add_closure_dep cds d a) = acc_symbols a
  ∧ ∀ a', acc_le (add_closure_dep cds d a) a' ->
      closure_dep_facts (acc_graph a') cds d.
Proof.
  unfold add_closure_dep, closure_dep_facts.
  destruct (lookup (cld_closure_code_id d) cds) as [cd | ].
  - split; [ apply add_facts_steady | split; [ reflexivity | ] ].
    split; [ reflexivity | ].
    intros a' Hle. split; [ | split ].
    + apply facts_propagate. apply (add_facts_later _ _ _ _ Hle). left.
      reflexivity.
    + apply facts_constructor. apply (add_facts_later _ _ _ _ Hle). right.
      left. reflexivity.
    + intros w Hw. apply facts_constructor.
      apply (add_facts_later _ _ _ _ Hle). right. right.
      destruct (cd_unknown_arity_call_witnesses cd) as [ | w0 ws0];
        [ discriminate Hw | ].
      injection Hw as <-. left. reflexivity.
  - destruct (fresh a) as [w a1] eqn:E.
    pose proof (steady_snd fresh a w a1 fresh_steady E) as S.
    injection E as <- <-.
    split; [ exact (steady_trans _ _ _ S (add_facts_steady _ _)) | ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    intros a' Hle. exists (Node_made (acc_next a)). split.
    + apply facts_any_source. apply (add_facts_later _ _ _ _ Hle). left.
      reflexivity.
    + apply facts_constructor. apply (add_facts_later _ _ _ _ Hle). right.
      left. reflexivity.
Qed.

Lemma resolve_delayed_deps_spec (unit_symbols : list symbol) (a : acc) :
  steady a (resolve_delayed_deps unit_symbols a)
  ∧ acc_closure_deps (resolve_delayed_deps unit_symbols a)
    = acc_closure_deps a
  ∧ acc_symbols (resolve_delayed_deps unit_symbols a) = acc_symbols a
  ∧ (∀ d : closure_dep,
       d ∈ acc_closure_deps a ->
       closure_dep_facts (acc_graph (resolve_delayed_deps unit_symbols a))
         (acc_code_deps a) d)
  ∧ ∀ s : symbol,
      s ∈ acc_symbols a ->
      among s unit_symbols = false ->
      Fact_any_source (Node_symbol s)
      ∈ acc_facts (resolve_delayed_deps unit_symbols a).
Proof.
  unfold resolve_delayed_deps.
  set (a1 := add_facts _ a).
  assert (Hfold : ∀ (l : list closure_dep) (b : acc),
            acc_code_deps b = acc_code_deps a ->
            steady b (List.fold_left (fun b d =>
                        add_closure_dep (acc_code_deps a) d b) l b)
            ∧ acc_closure_deps (List.fold_left (fun b d =>
                add_closure_dep (acc_code_deps a) d b) l b)
              = acc_closure_deps b
            ∧ acc_symbols (List.fold_left (fun b d =>
                add_closure_dep (acc_code_deps a) d b) l b) = acc_symbols b
            ∧ ∀ a', acc_le (List.fold_left (fun b d =>
                      add_closure_dep (acc_code_deps a) d b) l b) a' ->
                ∀ d, d ∈ l -> closure_dep_facts (acc_graph a')
                                (acc_code_deps a) d).
  { induction l as [ | d l IH]; intros b Hb; simpl.
    - split; [ apply steady_refl | split; [ reflexivity | ] ].
      split; [ reflexivity | intros ? ? ? [] ].
    - destruct (add_closure_dep_spec (acc_code_deps a) d b)
        as (S & Hdeps & Hsyms & Hfacts).
      destruct (IH _ (eq_trans (proj2 S) Hb)) as (S' & Hdeps' & Hsyms' & HP).
      split; [ exact (steady_trans _ _ _ S S') | ].
      split; [ congruence | split; [ congruence | ] ].
      intros a' Hle d' [<- | Hin].
      + apply Hfacts. exact (acc_le_trans _ _ _ (proj1 S') Hle).
      + exact (HP a' Hle d' Hin). }
  destruct (Hfold (acc_closure_deps a1) a1 eq_refl)
    as (S & Hdeps & Hsyms & HP).
  set (a2 := List.fold_left _ _ a1) in *.
  split; [ exact (steady_trans _ _ _ (add_facts_steady _ _)
                    (steady_trans _ _ _ S (add_facts_steady _ _))) | ].
  split; [ simpl; rewrite Hdeps; reflexivity | ].
  split; [ simpl; rewrite Hsyms; reflexivity | ].
  split.
  - intros d Hd. apply (HP _ (proj1 (add_facts_steady _ _)) d). exact Hd.
  - intros s Hin Hunit. simpl. apply in_or_app. left.
    apply in_map_iff. exists s. split; [ reflexivity | ].
    apply among_In. rewrite among_without, among_dedup, Hunit.
    rewrite andb_true_r. apply among_In. rewrite Hsyms. exact Hin.
Qed.

(* The symbols and the code ids that a term binds. *)
Lemma static_names_edges (G : graph) (noted : symbol -> Prop)
    (deps : closure_dep -> Prop) (cds : finite_map code_dep)
    (keep : value_slot -> bool) (codes : list (code_id * code))
    (symbols : list symbol) (slots_read heap_prims : Prop) :
  (∀ e : expr,
     ∀ (ctx : option code_id) (ce : finite_map (list node)),
       expr_edges G noted deps cds keep codes symbols slots_read heap_prims
         ctx ce e ->
       (∀ s : symbol, s ∈ static_names symbols_of_binding e ->
          ∃ const : static_const,
            binding_edges G noted deps cds keep codes symbols slots_read
              heap_prims
              (Bind_symbol s const))
       ∧ ∀ id : code_id, id ∈ static_names code_ids_of_binding e ->
           ∃ c : code,
             binding_edges G noted deps cds keep codes symbols slots_read
              heap_prims
               (Bind_code id c))
  ∧ (∀ h : continuation_handler,
       ∀ (ctx : option code_id) (ce : finite_map (list node)),
         handler_edges G noted deps cds keep codes symbols slots_read
           heap_prims ctx ce h ->
         (∀ s : symbol, s ∈ handler_static_names symbols_of_binding h ->
            ∃ const : static_const,
              binding_edges G noted deps cds keep codes symbols slots_read
              heap_prims
                (Bind_symbol s const))
         ∧ ∀ id : code_id, id ∈ handler_static_names code_ids_of_binding h ->
             ∃ c : code,
               binding_edges G noted deps cds keep codes symbols slots_read
              heap_prims
                 (Bind_code id c)).
Proof.
  apply expr_handler_ind; simpl.
  - intros x kind d body IH ctx ce [_ He]. exact (IH ctx ce He).
  - intros bindings body _ IH ctx ce [Hbindings He].
    destruct (IH ctx ce He) as [IHs IHc]. split.
    + intros s Hin. apply in_app_or in Hin as [Hin | Hin]; [ | exact (IHs s
      Hin) ].
      apply in_flat_map in Hin as (b & Hb & Hs).
      pose proof (proj1 (every_in _ _ _) Hbindings b Hb) as Hbe.
      destruct b as [ | | s' const]; simpl in Hs; try contradiction.
      destruct Hs as [<- | []]. exists const. exact Hbe.
    + intros id Hin. apply in_app_or in Hin as [Hin | Hin];
        [ | exact (IHc id Hin) ].
      apply in_flat_map in Hin as (b & Hb & Hs).
      pose proof (proj1 (every_in _ _ _) Hbindings b Hb) as Hbe.
      destruct b as [id' c | | ]; simpl in Hs; try contradiction.
      destruct Hs as [<- | []]. exists c. exact Hbe.
  - intros k h body IHh IH ctx ce [Hh He].
    destruct (IHh ctx ce Hh) as [IHhs IHhc].
    destruct (IH _ _ He) as [IHs IHc].
    split; intros n Hin; apply in_app_or in Hin as [Hin | Hin]; auto.
  - intros handlers body IHh IH ctx ce [Hhandlers He].
    destruct (IH _ _ He) as [IHs IHc].
    split; intros n Hin; apply in_app_or in Hin as [Hin | Hin]; auto;
      apply in_flat_map in Hin as ([k h] & Hkh & Hn);
      pose proof (proj1 (every_in _ _ _) Hhandlers _ Hkh) as Hh;
      destruct (IHh k h Hkh _ _ Hh) as [IHhs IHhc]; auto.
  - intros ap ctx ce _. split; intros ? [].
  - intros ac ctx ce _. split; intros ? [].
  - intros scrutinee arms ctx ce _. split; intros ? [].
  - intros ctx ce _. split; intros ? [].
  - intros params metadata handler IH ctx ce He. exact (IH ctx ce He).
Qed.

(* The accumulator and the environment from which [traverse_unit]
   traverses the unit. *)
Definition initial_acc : acc :=
  {| acc_next := 3;
     acc_facts :=
       [Fact_any_usage (Node_made 2); Fact_any_source Node_all_constants];
     acc_code_deps := []; acc_apply_deps := []; acc_closure_deps := [];
     acc_symbols := [] |}.

Definition initial_env (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) : traverse_env :=
  {| env_conts := unit_conts u;
     env_current_code_id := None;
     env_should_preserve_direct_calls := should_preserve_of_flag flag;
     env_value_slots_to_keep := projected_value_slots e;
     env_unit_code_ids := static_names all_code_ids_of_binding e;
     env_set_closures := set_closures e |}.

Lemma traverse_unit_eq (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  traverse_unit flag u e
  = facts_of_list
      (acc_facts
         (resolve_delayed_deps (static_names symbols_of_binding e)
            (add_fact (Fact_any_usage (Node_made 1))
               (traverse (initial_env flag u e) e initial_acc)))).
Proof. reflexivity. Qed.

(* What the proofs need of the graph of a unit of the fragment. *)
Lemma traverse_unit_facts_gen (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (slots_read : Prop) (hb : bool) :
  flow_fragment hb (unit_codes e) (static_names symbols_of_binding e) e
  && distinct (static_names code_ids_of_binding e) = true ->
  slots_read ∨ projection_free e = true ->
  ∃ (noted : symbol -> Prop) (deps : closure_dep -> Prop)
    (cds : finite_map code_dep),
    expr_edges (graph_of_facts (traverse_unit flag u e)) noted deps cds
      (unit_kept e) (unit_codes e) (static_names symbols_of_binding e)
      slots_read (if hb then True else False) None (unit_conts u) e
    ∧ g_any_source (graph_of_facts (traverse_unit flag u e)) Node_all_constants
    ∧ (∀ s : symbol,
         noted s ->
         among s (static_names symbols_of_binding e) = false ->
         g_any_source (graph_of_facts (traverse_unit flag u e))
           (Node_symbol s))
    ∧ (∀ (k : continuation) (p : node),
         p ∈ unit_cont_nodes u k ->
         g_any_usage (graph_of_facts (traverse_unit flag u e)) p)
    ∧ (∀ d : closure_dep,
         deps d ->
         closure_dep_facts (graph_of_facts (traverse_unit flag u e)) cds d)
    ∧ NoDup (map fst cds)
    ∧ (∀ id : code_id,
         id ∈ map fst cds -> id ∈ static_names code_ids_of_binding e).
Proof.
  intros Hfrag Hslots. apply andb_true_iff in Hfrag as [Hfrag Hdist].
  rewrite traverse_unit_eq.
  destruct (proj1 (traverse_makes (unit_codes e)
                     (static_names symbols_of_binding e)
                     (projected_value_slots e) slots_read hb) e Hfrag Hdist
              (fun w Hw => proj2 (among_In w _) Hw) Hslots (fun x Hx => Hx)
              (initial_env flag u e) eq_refl initial_acc)
    as [[Hle4 (l & Hl & Hperm)] Hedges].
  set (a4 := traverse (initial_env flag u e) e initial_acc) in *.
  set (a5 := add_fact (Fact_any_usage (Node_made 1)) a4).
  destruct (resolve_delayed_deps_spec (static_names symbols_of_binding e) a5)
    as (S6 & Hdeps6 & Hsyms6 & Hclosures & Himported).
  set (a6 := resolve_delayed_deps _ a5) in *.
  pose proof (steady_trans _ _ _ (add_facts_steady _ a4) S6) as S46.
  exists (acc_noted a6), (acc_deps a6), (acc_code_deps a6).
  assert (Hcds : acc_code_deps a6 = l).
  { rewrite (proj2 S46), Hl, app_nil_r. reflexivity. }
  split; [ | split; [ | split; [ | split; [ | split; [ | split ] ] ] ] ].
  - exact (Hedges a6 (proj1 S46)).
  - apply facts_any_source. apply (proj1 S46), Hle4. right. left.
    reflexivity.
  - intros s Hs Hunit. apply facts_any_source.
    apply Himported; [ | exact Hunit ].
    unfold acc_noted in Hs. rewrite Hsyms6 in Hs. exact Hs.
  - intros k p Hp. unfold unit_cont_nodes, unit_conts in Hp.
    simpl in Hp. apply facts_any_usage.
    destruct (Pos.eqb k (unit_exn u));
      [ | destruct (Pos.eqb k (unit_return u)) ];
      simpl in Hp; [ | | contradiction ]; destruct Hp as [<- | []].
    + apply (proj1 S46), Hle4. left. reflexivity.
    + apply (proj1 S6). left. reflexivity.
  - intros d Hd. unfold acc_deps in Hd. rewrite Hdeps6 in Hd.
    pose proof (Hclosures d Hd) as H. rewrite <- (proj2 S6) in H.
    exact H.
  - rewrite Hcds. apply (Permutation_NoDup (Permutation_sym Hperm)).
    apply distinct_NoDup. exact Hdist.
  - intros id Hid. rewrite Hcds in Hid.
    exact (Permutation_in _ Hperm Hid).
Qed.

(* The facts of a unit of the fragment without the primitives of the
   heap, and with them. *)
Lemma traverse_unit_facts (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (slots_read : Prop) :
  in_flow_fragment e = true ->
  slots_read ∨ projection_free e = true ->
  ∃ (noted : symbol -> Prop) (deps : closure_dep -> Prop)
    (cds : finite_map code_dep),
    expr_edges (graph_of_facts (traverse_unit flag u e)) noted deps cds
      (unit_kept e) (unit_codes e) (static_names symbols_of_binding e)
      slots_read False None (unit_conts u) e
    ∧ g_any_source (graph_of_facts (traverse_unit flag u e)) Node_all_constants
    ∧ (∀ s : symbol,
         noted s ->
         among s (static_names symbols_of_binding e) = false ->
         g_any_source (graph_of_facts (traverse_unit flag u e))
           (Node_symbol s))
    ∧ (∀ (k : continuation) (p : node),
         p ∈ unit_cont_nodes u k ->
         g_any_usage (graph_of_facts (traverse_unit flag u e)) p)
    ∧ (∀ d : closure_dep,
         deps d ->
         closure_dep_facts (graph_of_facts (traverse_unit flag u e)) cds d)
    ∧ NoDup (map fst cds)
    ∧ (∀ id : code_id,
         id ∈ map fst cds -> id ∈ static_names code_ids_of_binding e).
Proof. exact (traverse_unit_facts_gen flag u e slots_read false). Qed.

Lemma traverse_unit_facts_heap (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (slots_read : Prop) :
  in_heap_fragment e = true ->
  slots_read ∨ projection_free e = true ->
  ∃ (noted : symbol -> Prop) (deps : closure_dep -> Prop)
    (cds : finite_map code_dep),
    expr_edges (graph_of_facts (traverse_unit flag u e)) noted deps cds
      (unit_kept e) (unit_codes e) (static_names symbols_of_binding e)
      slots_read True None (unit_conts u) e
    ∧ g_any_source (graph_of_facts (traverse_unit flag u e)) Node_all_constants
    ∧ (∀ s : symbol,
         noted s ->
         among s (static_names symbols_of_binding e) = false ->
         g_any_source (graph_of_facts (traverse_unit flag u e))
           (Node_symbol s))
    ∧ (∀ (k : continuation) (p : node),
         p ∈ unit_cont_nodes u k ->
         g_any_usage (graph_of_facts (traverse_unit flag u e)) p)
    ∧ (∀ d : closure_dep,
         deps d ->
         closure_dep_facts (graph_of_facts (traverse_unit flag u e)) cds d)
    ∧ NoDup (map fst cds)
    ∧ (∀ id : code_id,
         id ∈ map fst cds -> id ∈ static_names code_ids_of_binding e).
Proof. exact (traverse_unit_facts_gen flag u e slots_read true). Qed.

(* ======================================================================== *)
(* The primitives of the default case                                       *)
(* ======================================================================== *)

(* Solves a goal whose first disjunct is that [v] is a constant, given
   that a computation that makes a constant is [Some v]. *)
Ltac const_value H :=
  unfold option_map, array_length_of, eval_is_int, eval_is_null,
    eval_reinterpret_64_bit_word, eval_string_load in H;
  repeat
    match type of H with
    | context [match ?o with _ => _ end] => destruct o; try discriminate H
    end;
  injection H as <-; left; eexists; reflexivity.

(* What a primitive that [traverse_prim] does not read in a way of its
   own computes: a constant, its one argument (the identity), an element
   of an immutable array that is its first argument, or an immutable
   array of its arguments, each of which is bound if it is a symbol. *)
Lemma default_prim_result (statics : static_env) (p : prim)
    (args : list simple) (vs : list value) (v : value) :
  prim_classified p args = false ->
  length vs = length args ->
  eval_prim statics p vs = Some v ->
  (∃ c : const, v = Value_const c)
  ∨ vs = [v]
  ∨ (∃ (arr idx : value) (elements : list value),
       vs = [arr; idx] ∧ array_of statics arr = Some elements
       ∧ v ∈ elements)
  ∨ (v = Value_immutable_array vs
     ∧ ∀ w : value, w ∈ vs -> bound_if_symbol statics w = true).
Proof.
  intros Hcl Hlen Heval.
  destruct (eval_prim_arity _ _ _ _ Heval)
    as [(p' & v1 & -> & -> & Hu) | [(p' & v1 & v2 & -> & -> & Hb)
                                  | (p' & -> & Hv)]].
  - destruct args as [ | s1 [ | s2 args]]; simpl in Hlen; try discriminate Hlen.
    destruct p' as [kind mut fld | [ | ] | | | ak | | ok | ik op | src dst
                   | | bn | bn | | | r];
      simpl in Hcl; try discriminate Hcl; cbn [eval_unary_primitive] in Hu;
      try (const_value Hu).
    (* [Opaque_identity] *)
    destruct (kind_eqb (kind_of_value v1) ok && bound_if_symbol statics v1);
      [ | discriminate Hu ].
    injection Hu as ->. right. left. reflexivity.
  - destruct p' as [ak lk mut' | width | op | ik op | ik op | ik beh];
      cbn [eval_binary_primitive] in Hb; try (const_value Hb).
    (* [Array_load] *)
    destruct (array_of statics v1) as [elements | ] eqn:Harr;
      [ | discriminate Hb ].
    destruct (array_loaded_element _ _ _ _ Hb) as [Hin _].
    right. right. left. exists v1, v2, elements. auto.
  - destruct p' as [bk | ak]; [ discriminate Hcl | ].
    cbn [eval_variadic_primitive] in Hv.
    destruct (array_made ak vs) as [elements | ] eqn:Hmade;
      [ | discriminate Hv ].
    destruct (array_made_of_arguments _ _ _ Hmade) as [-> _].
    destruct (forallb (bound_if_symbol statics) vs) eqn:Hbound;
      [ | discriminate Hv ].
    injection Hv as <-. right. right. right. split; [ reflexivity | ].
    rewrite forallb_forall in Hbound. exact Hbound.
Qed.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma eval_simples_nth (vars : env) (args : list simple) (vs : list value)
    (i : nat) (v : value) :
  eval_simples vars args = Some vs ->
  nth_error vs i = Some v ->
  ∃ s : simple, nth_error args i = Some s ∧ eval_simple vars s = Some v.
Proof.
  revert vs i.
  induction args as [ | s args IH]; intros vs i Hev Hnth; simpl in Hev.
  - injection Hev as <-. destruct i; discriminate.
  - destruct (eval_simple vars s) as [w | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
    injection Hev as <-. destruct i as [ | i]; simpl in Hnth |- *.
    + injection Hnth as <-. exists s. auto.
    + exact (IH ws i eq_refl Hnth).
Qed.

Lemma eval_simples_in (vars : env) (args : list simple) (vs : list value)
    (v : value) :
  eval_simples vars args = Some vs -> v ∈ vs ->
  ∃ s : simple, s ∈ args ∧ eval_simple vars s = Some v.
Proof.
  intros Hev Hin. apply In_nth_error in Hin as [i Hi].
  destruct (eval_simples_nth _ _ _ _ _ Hev Hi) as (s & Hs & Hv).
  exists s. split; [ exact (nth_error_In _ _ Hs) | exact Hv ].
Qed.

Lemma eval_simples_length (vars : env) (args : list simple) (vs : list value) :
  eval_simples vars args = Some vs -> length vs = length args.
Proof.
  revert vs. induction args as [ | s args IH]; intros vs Hev; simpl in Hev.
  - injection Hev as <-. reflexivity.
  - destruct (eval_simple vars s); [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ]; [ | discriminate ].
    injection Hev as <-. simpl. f_equal. exact (IH ws eq_refl).
Qed.

Lemma fields_from_intro (P : nat -> value -> Prop) (k : nat)
    (l : list value) :
  (∀ (i : nat) (v : value), nth_error l i = Some v -> P (k + i) v) ->
  fields_from P k l.
Proof.
  revert k. induction l as [ | v l IH]; intros k H; simpl; [ exact I | ].
  split.
  - replace k with (k + 0) by lia. apply H. reflexivity.
  - apply IH. intros i w Hw. replace (S k + i) with (k + S i) by lia.
    apply H. exact Hw.
Qed.

Lemma fields_from_elim (P : nat -> value -> Prop) (k : nat) (l : list value)
    (i : nat) (v : value) :
  fields_from P k l -> nth_error l i = Some v -> P (k + i) v.
Proof.
  revert k i. induction l as [ | w l IH]; intros k i H Hnth;
    [ destruct i; discriminate | ].
  destruct H as [Hw Hl]. destruct i as [ | i]; simpl in Hnth.
  - injection Hnth as <-. replace (k + 0) with k by lia. exact Hw.
  - replace (k + S i) with (S k + i) by lia. exact (IH _ _ Hl Hnth).
Qed.

Lemma bind_params_lookup (params : list variable) (vs : list value)
    (vars vars' : env) (x : variable) (v : value) :
  bind_params params vs vars = Some vars' ->
  lookup x vars' = Some v ->
  lookup x vars = Some v
  ∨ ∃ i : nat, nth_error params i = Some x ∧ nth_error vs i = Some v.
Proof.
  revert vs vars. induction params as [ | p params IH];
    intros [ | w vs] vars Hbind Hlookup; simpl in Hbind; try discriminate.
  - injection Hbind as <-. left. exact Hlookup.
  - destruct (IH vs _ Hbind Hlookup) as [Hl | (i & Hp & Hv)].
    + simpl in Hl. destruct (Pos.eqb_spec x p) as [-> | Hne].
      * injection Hl as ->. right. exists 0. auto.
      * left. exact Hl.
    + right. exists (S i). auto.
Qed.

Lemma nth_error_combine {A B : Type} (l : list A) (l' : list B) (i : nat)
    (a : A) (b : B) :
  nth_error l i = Some a -> nth_error l' i = Some b -> (a, b) ∈ combine l l'.
Proof.
  revert l' i. induction l as [ | x l IH]; intros [ | y l'] i Ha Hb;
    try (destruct i; discriminate).
  destruct i as [ | i]; simpl in Ha, Hb |- *.
  - injection Ha as <-. injection Hb as <-. left. reflexivity.
  - right. exact (IH l' i Ha Hb).
Qed.

(* The free names of the arguments of a primitive are those of its
   variables and symbols. *)
Lemma free_names_var (args : list simple) (y : variable) :
  Var y ∈ args -> Node_variable y ∈ free_names_of_simples args.
Proof.
  intros Hin. apply in_flat_map. exists (Var y). split; [ exact Hin | ].
  left. reflexivity.
Qed.

Lemma free_names_symbol (args : list simple) (s : symbol) :
  Symbol s ∈ args -> Node_symbol s ∈ free_names_of_simples args.
Proof.
  intros Hin. apply in_flat_map. exists (Symbol s). split; [ exact Hin | ].
  left. reflexivity.
Qed.

(* Two boxable kinds of one number are one. *)
Lemma is_number_of_kind_unique (bn bn' : boxable_number) (c : const) :
  is_number_of_kind bn c = true -> is_number_of_kind bn' c = true -> bn = bn'.
Proof. destruct bn, bn', c; simpl; congruence. Qed.


(* ======================================================================== *)
(* The values of a run are where the solution says                          *)
(* ======================================================================== *)

(* What the closure dep of the closure at [b], of the code [id], gives
   it. *)
Definition called_by (G : graph) (cds : finite_map code_dep) (b : node)
    (id : code_id) : Prop :=
  closure_dep_facts G cds
    {| cld_let_bound_name_of_the_closure := b; cld_closure_code_id := id |}.

Lemma eval_value_slots_lookup (vars : env)
    (value_slots : list (value_slot * simple))
    (vs : list (value_slot * value)) (w : value_slot) (v : value) :
  eval_value_slots vars value_slots = Some vs ->
  lookup w vs = Some v ->
  ∃ s : simple, lookup w value_slots = Some s ∧ eval_simple vars s = Some v.
Proof.
  revert vs. induction value_slots as [ | [w' s'] value_slots IH];
    intros vs Hev Hl; simpl in Hev.
  - injection Hev as <-. discriminate Hl.
  - destruct (eval_simple vars s') as [v' | ] eqn:Hs'; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs' | ] eqn:Hvs';
      [ | discriminate ].
    injection Hev as <-. simpl in Hl |- *.
    destruct (Pos.eqb w w').
    + injection Hl as <-. exists s'. auto.
    + exact (IH vs' eq_refl Hl).
Qed.

(* ------------------------------------------------------------------------ *)
(* Groups of arguments                                                      *)
(* ------------------------------------------------------------------------ *)

Lemma nth_error_firstn_lt {A : Type} (l : list A) (n i : nat) :
  i < n -> nth_error (firstn n l) i = nth_error l i.
Proof.
  revert l i. induction n as [ | n IH]; intros [ | a l] [ | i] H; simpl;
    try lia; try reflexivity.
  apply IH. lia.
Qed.

Lemma nth_error_skipn_add {A : Type} (l : list A) (n i : nat) :
  nth_error (skipn n l) i = nth_error l (n + i).
Proof.
  revert l. induction n as [ | n IH]; intros l; [ reflexivity | ].
  destruct l as [ | a l]; simpl; [ destruct i; reflexivity | apply IH ].
Qed.

Lemma group_position_lt (sizes : list nat) (i g k : nat) :
  group_position sizes i = Some (g, k) ->
  g < length sizes ∧ i < list_sum sizes.
Proof.
  revert i g. induction sizes as [ | size sizes IH]; intros i g H;
    simpl in H; [ discriminate | ].
  destruct (Nat.ltb_spec i size) as [Hi | Hi].
  - injection H as <- <-. simpl. lia.
  - destruct (group_position sizes (i - size)) as [[g' k'] | ] eqn:E;
      [ | discriminate ].
    injection H as <- <-. destruct (IH _ _ E) as [Hg Hs]. simpl. lia.
Qed.

Lemma group_position_some (sizes : list nat) (i : nat) :
  i < list_sum sizes -> ∃ g k : nat, group_position sizes i = Some (g, k).
Proof.
  revert i. induction sizes as [ | size sizes IH]; intros i H; simpl in *;
    [ lia | ].
  destruct (Nat.ltb_spec i size) as [Hi | Hi]; [ eauto | ].
  destruct (IH (i - size) ltac:(lia)) as (g & k & E). rewrite E. eauto.
Qed.

Lemma group_position_app_l (s1 s2 : list nat) (i : nat) :
  i < list_sum s1 -> group_position (s1 ++ s2) i = group_position s1 i.
Proof.
  revert i. induction s1 as [ | size s1 IH]; intros i H; simpl in *; [ lia | ].
  destruct (Nat.ltb_spec i size); [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma group_position_app_r (s1 s2 : list nat) (i : nat) :
  list_sum s1 <= i ->
  group_position (s1 ++ s2) i
  = match group_position s2 (i - list_sum s1) with
    | Some (g, k) => Some (length s1 + g, k)
    | None => None
    end.
Proof.
  revert i. induction s1 as [ | size s1 IH]; intros i H; simpl in *.
  - rewrite Nat.sub_0_r.
    destruct (group_position s2 i) as [[g k] | ]; reflexivity.
  - destruct (Nat.ltb_spec i size); [ lia | ].
    rewrite IH by lia.
    replace (i - size - list_sum s1) with (i - (size + list_sum s1)) by lia.
    destruct (group_position s2 _) as [[g k] | ]; reflexivity.
Qed.

Lemma group_by_sizes_nth {A : Type} (sizes : list nat) (l : list A)
    (i g k : nat) (x : A) :
  group_position sizes i = Some (g, k) -> nth_error l i = Some x ->
  ∃ grp : list A,
    nth_error (group_by_sizes sizes l) g = Some grp ∧ nth_error grp k = Some x.
Proof.
  revert l i g. induction sizes as [ | size sizes IH]; intros l i g Hpos Hx;
    simpl in Hpos; [ discriminate | ].
  simpl. destruct (Nat.ltb_spec i size) as [Hi | Hi].
  - injection Hpos as <- <-. exists (firstn size l).
    split; [ reflexivity | ]. rewrite nth_error_firstn_lt by exact Hi.
    exact Hx.
  - destruct (group_position sizes (i - size)) as [[g' k'] | ] eqn:E;
      [ | discriminate ].
    injection Hpos as <- <-.
    apply (IH (skipn size l) (i - size)); [ exact E | ].
    rewrite nth_error_skipn_add. replace (size + (i - size)) with i by lia.
    exact Hx.
Qed.

Lemma length_group_by_sizes {A : Type} (sizes : list nat) (l : list A) :
  length (group_by_sizes sizes l) = length sizes.
Proof.
  revert l. induction sizes as [ | size sizes IH]; intros l; simpl;
    [ reflexivity | f_equal; apply IH ].
Qed.

Lemma arity_sizes_app (a1 a2 : arity) :
  arity_sizes (a1 ++ a2) = arity_sizes a1 ++ arity_sizes a2.
Proof. unfold arity_sizes, unarize_per_parameter. rewrite !map_app. reflexivity.
Qed.

Lemma length_arity_sizes (a : arity) : length (arity_sizes a) = length a.
Proof. unfold arity_sizes, unarize_per_parameter. rewrite !length_map.
  reflexivity. Qed.

Lemma sum_arity_sizes (a : arity) :
  list_sum (arity_sizes a) = length (unarize a).
Proof.
  unfold arity_sizes, unarize_per_parameter, unarize.
  induction a as [ | c a IH]; simpl; [ reflexivity | ].
  rewrite length_app, IH. reflexivity.
Qed.

Lemma arity_sizes_firstn (k : nat) (a : arity) :
  arity_sizes (firstn k a) = firstn k (arity_sizes a).
Proof.
  unfold arity_sizes, unarize_per_parameter.
  rewrite <- !firstn_map. reflexivity.
Qed.

Lemma arity_sizes_skipn (k : nat) (a : arity) :
  arity_sizes (skipn k a) = skipn k (arity_sizes a).
Proof.
  unfold arity_sizes, unarize_per_parameter.
  rewrite <- !skipn_map. reflexivity.
Qed.

Lemma group_by_parameter_groups {A : Type} (a : arity) (l : list A) :
  a <> [] ->
  at_least_one_group (group_by_parameter a l)
  = group_by_sizes (arity_sizes a) l.
Proof. destruct a; [ contradiction | reflexivity ]. Qed.

(* ------------------------------------------------------------------------ *)
(* Partial applications                                                     *)
(* ------------------------------------------------------------------------ *)

(* What the run keeps of each partial application: that it has fewer
   groups of arguments than its code has parameters, and that the code
   is not tupled. A call made it so, and the static environment only
   grows. *)
Definition unsaturated (statics : static_env) (closure : value) (a : arity)
    : Prop :=
  ∃ (id : code_id) (slots : list (value_slot * value)) (c : code),
    closure_of statics closure = Some (id, slots)
    ∧ lookup id (static_code statics) = Some c
    ∧ length a < length (code_params_arity c)
    ∧ code_is_tupled c = false.

Fixpoint pa_ok (statics : static_env) (v : value) {struct v} : Prop :=
  match v with
  | Value_closure _ slots | Value_closure_of_set _ _ slots =>
    every (fun binding => pa_ok statics (snd binding)) slots
  | Value_block _ fields => every (fun field => pa_ok statics field) fields
  | Value_immutable_array elements =>
    every (fun element => pa_ok statics element) elements
  | Value_partial_application closure args a =>
    unsaturated statics closure a ∧ pa_ok statics closure
    ∧ every (fun arg => pa_ok statics arg) args
  | Value_const _ | Value_symbol _ | Value_boxed_number _ | Value_location _ =>
    True
  end.

Definition static_pa_ok (statics : static_env)
    (allocated : statically_allocated_value) : Prop :=
  match allocated with
  | Static_closure _ slots | Static_closure_of_set _ _ _ slots =>
    every (fun binding => pa_ok statics (snd binding)) slots
  | Static_block _ fields => every (pa_ok statics) fields
  | Static_boxed_number contents => pa_ok statics contents
  | Static_immutable_array elements => every (pa_ok statics) elements
  | Static_string _ | Static_imported _ => True
  end.

Definition statics_pa_ok (statics : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    static_pa_ok statics allocated.

Definition env_pa_ok (statics : static_env) (vars : env) : Prop :=
  ∀ (x : variable) (v : value), lookup x vars = Some v -> pa_ok statics v.

Inductive cont_pa_ok (statics : static_env) : cont_value -> Prop :=
  | Closure_pa (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    env_pa_ok statics vars ->
    (∀ (k : continuation) (c : cont_value),
       (k, c) ∈ conts -> cont_pa_ok statics c) ->
    cont_pa_ok statics (Closure h vars conts recursive)
  | Outer_pa (k : continuation) : cont_pa_ok statics (Outer k)
  | Extra_pa (c : cont_value) (extra : list value) :
    cont_pa_ok statics c -> every (pa_ok statics) extra ->
    cont_pa_ok statics (Extra_args c extra)
  | Over_pa (remaining : list value) (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    every (pa_ok statics) remaining ->
    cont_pa_ok statics return_to -> cont_pa_ok statics exn_to ->
    cont_pa_ok statics
      (Over_application remaining remaining_arity return_arity return_to
         exn_to).

Definition conts_pa_ok (statics : static_env) (conts : cont_env) : Prop :=
  ∀ (k : continuation) (c : cont_value),
    (k, c) ∈ conts -> cont_pa_ok statics c.

Definition statics_le (statics statics' : static_env) : Prop :=
  (∀ (s : symbol) (allocated : statically_allocated_value),
     lookup s (static_values statics) = Some allocated ->
     lookup s (static_values statics') = Some allocated)
  ∧ ∀ (id : code_id) (c : code),
      lookup id (static_code statics) = Some c ->
      lookup id (static_code statics') = Some c.

Lemma closure_of_le (statics statics' : static_env) (v : value)
    (id : code_id) (slots : list (value_slot * value)) :
  statics_le statics statics' ->
  closure_of statics v = Some (id, slots) ->
  closure_of statics' v = Some (id, slots).
Proof.
  intros [Hv _] H. destruct v; simpl in *; try exact H.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:E;
    [ | discriminate ].
  rewrite (Hv _ _ E). exact H.
Qed.

Lemma pa_ok_le (statics statics' : static_env) (v : value) :
  statics_le statics statics' -> pa_ok statics v -> pa_ok statics' v.
Proof.
  intros Hle. pattern v. apply FlambdaHeapProofs.value_induction; clear v;
    simpl.
  - auto.
  - intros code slots IH. apply every_impl. intros b Hb.
    rewrite Forall_forall in IH. exact (IH b Hb).
  - intros slot decls slots IH. apply every_impl. intros b Hb.
    rewrite Forall_forall in IH. exact (IH b Hb).
  - auto.
  - intros tag fields IH. apply every_impl. intros f Hf.
    rewrite Forall_forall in IH. exact (IH f Hf).
  - intros elements IH. apply every_impl. intros f Hf.
    rewrite Forall_forall in IH. exact (IH f Hf).
  - auto.
  - auto.
  - intros closure args a IHc IHargs (Hun & Hcl & Hargs).
    split; [ | split; [ exact (IHc Hcl) | ] ].
    + destruct Hun as (id & slots & c & Hof & Hcode & Hlt & Hnt).
      exists id, slots, c.
      split; [ exact (closure_of_le _ _ _ _ _ Hle Hof) | ].
      split; [ exact (proj2 Hle _ _ Hcode) | split; [ exact Hlt | exact Hnt ]
      ].
    + revert Hargs. apply every_impl. intros x Hx.
      rewrite Forall_forall in IHargs. exact (IHargs x Hx).
Qed.

Lemma every_pa_ok_le (statics statics' : static_env) (l : list value) :
  statics_le statics statics' ->
  every (pa_ok statics) l -> every (pa_ok statics') l.
Proof.
  intros Hle. apply every_impl. intros v _. exact (pa_ok_le _ _ _ Hle).
Qed.

Lemma env_pa_ok_le (statics statics' : static_env) (vars : env) :
  statics_le statics statics' ->
  env_pa_ok statics vars -> env_pa_ok statics' vars.
Proof.
  intros Hle H x v Hx. exact (pa_ok_le _ _ _ Hle (H x v Hx)).
Qed.

Lemma cont_pa_ok_le (statics statics' : static_env) (c : cont_value) :
  statics_le statics statics' ->
  cont_pa_ok statics c -> cont_pa_ok statics' c.
Proof.
  intros Hle H. induction H as [h vars conts recursive Hvars _ IH | k
    | c extra _ IH Hextra | remaining remaining_arity return_arity return_to
        exn_to Hrem _ IHret _ IHexn].
  - apply Closure_pa; [ exact (env_pa_ok_le _ _ _ Hle Hvars) | exact IH ].
  - apply Outer_pa.
  - apply Extra_pa; [ exact IH | exact (every_pa_ok_le _ _ _ Hle Hextra) ].
  - apply Over_pa; [ exact (every_pa_ok_le _ _ _ Hle Hrem) | exact IHret
      | exact IHexn ].
Qed.

Lemma static_pa_ok_le (statics statics' : static_env)
    (allocated : statically_allocated_value) :
  statics_le statics statics' ->
  static_pa_ok statics allocated -> static_pa_ok statics' allocated.
Proof.
  intros Hle. destruct allocated; simpl; try auto;
    try (apply every_impl; intros b _; exact (pa_ok_le _ _ _ Hle)).
  apply pa_ok_le. exact Hle.
Qed.

Lemma eval_static_bindings_le (vars : env) (bindings : list static_binding)
    (statics statics' : static_env) :
  eval_static_bindings vars bindings statics = Some statics' ->
  statics_le statics statics'.
Proof.
  revert statics.
  induction bindings as [ | b bindings IH]; intros statics Heval;
    simpl in Heval.
  - injection Heval as <-. split; auto.
  - destruct b as [id c | id | s const].
    + destruct (lookup id (static_code statics)) eqn:Hl;
        [ discriminate Heval | ].
      destruct (IH _ Heval) as [Hv Hc']. split.
      * intros s a H. apply Hv. exact H.
      * intros id' c' H. apply Hc'. simpl.
        destruct (Pos.eqb_spec id' id) as [-> | _]; [ congruence | exact H ].
    + exact (IH _ Heval).
    + destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Heval | ].
      destruct (eval_static_const vars const) as [allocated | ];
        [ | discriminate Heval ].
      destruct (IH _ Heval) as [Hv Hc']. split.
      * intros s' a H. apply Hv. simpl.
        destruct (Pos.eqb_spec s' s) as [-> | _]; [ congruence | exact H ].
      * intros id' c' H. apply Hc'. exact H.
Qed.

(* The values of a value slot of a closure, of an environment with
   partial applications as the run keeps them. *)
Lemma closure_slots_pa_ok (statics : static_env) (v : value) (id : code_id)
    (slots : list (value_slot * value)) :
  pa_ok statics v -> statics_pa_ok statics ->
  closure_of statics v = Some (id, slots) ->
  every (fun binding => pa_ok statics (snd binding)) slots.
Proof.
  intros Hv Hst Hcl. destruct v; simpl in Hcl, Hv; try discriminate.
  - injection Hcl as <- <-. exact Hv.
  - destruct (lookup slot function_decls); [ | discriminate ].
    injection Hcl as <- <-. exact Hv.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:E;
      [ | discriminate ].
    specialize (Hst s allocated E).
    destruct allocated; try discriminate; simpl in Hst.
    + injection Hcl as <- <-. exact Hst.
    + destruct (lookup slot function_decls); [ | discriminate ].
      injection Hcl as <- <-. exact Hst.
Qed.

Lemma eval_value_slots_in (vars : env)
    (value_slots : list (value_slot * simple))
    (vs : list (value_slot * value)) (w : value_slot) (v : value) :
  eval_value_slots vars value_slots = Some vs ->
  (w, v) ∈ vs ->
  ∃ s : simple, (w, s) ∈ value_slots ∧ eval_simple vars s = Some v.
Proof.
  revert vs. induction value_slots as [ | [w' s'] value_slots IH];
    intros vs Hev Hin; simpl in Hev.
  - injection Hev as <-. destruct Hin.
  - destruct (eval_simple vars s') as [v' | ] eqn:Hs';
      [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs' | ] eqn:Hvs';
      [ | discriminate ].
    injection Hev as <-. destruct Hin as [[= <- <-] | Hin].
    + exists s'. split; [ left; reflexivity | exact Hs' ].
    + destruct (IH vs' eq_refl Hin) as (s & Hin' & Hs).
      exists s. split; [ right; exact Hin' | exact Hs ].
Qed.

Lemma simple_pa_ok (statics : static_env) (vars : env) (s : simple)
    (v : value) :
  env_pa_ok statics vars -> eval_simple vars s = Some v -> pa_ok statics v.
Proof.
  intros Hvars Hs. destruct s as [x | sym | c]; simpl in Hs.
  - exact (Hvars x v Hs).
  - injection Hs as <-. exact I.
  - injection Hs as <-. exact I.
Qed.

Lemma simples_pa_ok (statics : static_env) (vars : env) (l : list simple)
    (vs : list value) :
  env_pa_ok statics vars -> eval_simples vars l = Some vs ->
  every (pa_ok statics) vs.
Proof.
  intros Hvars Hvs. apply (every_in _ _ _). intros v Hv.
  destruct (eval_simples_in _ _ _ _ Hvs Hv) as (s & _ & Hs).
  exact (simple_pa_ok _ _ _ _ Hvars Hs).
Qed.

Lemma value_slots_pa_ok (statics : static_env) (vars : env)
    (value_slots : list (value_slot * simple))
    (vs : list (value_slot * value)) :
  env_pa_ok statics vars -> eval_value_slots vars value_slots = Some vs ->
  every (fun binding => pa_ok statics (snd binding)) vs.
Proof.
  intros Hvars Hvs. apply (every_in _ _ _). intros [w v] Hin.
  destruct (eval_value_slots_in _ _ _ _ _ Hvs Hin) as (s & _ & Hs).
  exact (simple_pa_ok _ _ _ _ Hvars Hs).
Qed.

(* The value of a [Let]. *)
Lemma named_pa_ok (statics : static_env) (vars : env) (d : named)
    (v : value) :
  env_pa_ok statics vars -> statics_pa_ok statics ->
  eval_named statics vars d = Some v -> pa_ok statics v.
Proof.
  intros Hvars Hst Hev.
  destruct d as [s | p args | code value_slots | slot decls value_slots
                | slot closure | from slot closure | from to_ closure];
    simpl in Hev.
  - exact (simple_pa_ok _ _ _ _ Hvars Hev).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hvs; [ | discriminate ].
    pose proof (simples_pa_ok _ _ _ _ Hvars Hvs) as Hvs_pa.
    apply (FlambdaPrimitiveProofs.eval_prim_preserves (pa_ok statics)
             statics p vs v); try exact Hev.
    + intros c. split; exact I.
    + intros tag fields. simpl. apply (every_in _ _ _).
    + intros elements. simpl. apply (every_in _ _ _).
    + intros s tag fields _ Hs w Hw. specialize (Hst s _ Hs). simpl in Hst.
      exact (proj1 (every_in _ _ _) Hst w Hw).
    + intros s elements _ Hs w Hw. specialize (Hst s _ Hs). simpl in Hst.
      exact (proj1 (every_in _ _ _) Hst w Hw).
    + exact (proj1 (every_in _ _ _) Hvs_pa).
  - destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    injection Hev as <-. exact (value_slots_pa_ok _ _ _ _ Hvars Hvs).
  - destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    injection Hev as <-. exact (value_slots_pa_ok _ _ _ _ Hvars Hvs).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hw; [ | discriminate ].
    destruct (closure_of statics w) as [[id slots] | ] eqn:Hcl;
      [ | discriminate ].
    pose proof (closure_slots_pa_ok _ _ _ _ (simple_pa_ok _ _ _ _ Hvars Hw)
                  Hst Hcl) as Hslots.
    simpl in Hev. apply lookup_in in Hev.
    exact (proj1 (every_in _ _ _) Hslots _ Hev).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hw; [ | discriminate ].
    pose proof (simple_pa_ok _ _ _ _ Hvars Hw) as Hwpa.
    unfold project_value_slot in Hev.
    destruct w; simpl in Hev, Hwpa; try discriminate.
    + destruct (Pos.eqb unnamed_function_slot from); [ | discriminate ].
      apply lookup_in in Hev. exact (proj1 (every_in _ _ _) Hwpa _ Hev).
    + destruct (lookup slot0 function_decls); simpl in Hev;
        [ | discriminate ].
      destruct (Pos.eqb slot0 from); [ | discriminate ].
      apply lookup_in in Hev. exact (proj1 (every_in _ _ _) Hwpa _ Hev).
    + destruct (lookup s (static_values statics)) as [allocated | ] eqn:E;
        [ | discriminate ].
      specialize (Hst s allocated E).
      destruct allocated; simpl in Hev, Hst; try discriminate.
      * destruct (Pos.eqb unnamed_function_slot from); [ | discriminate ].
        apply lookup_in in Hev. exact (proj1 (every_in _ _ _) Hst _ Hev).
      * destruct (lookup slot0 function_decls); simpl in Hev;
          [ | discriminate ].
        destruct (Pos.eqb slot0 from); [ | discriminate ].
        apply lookup_in in Hev. exact (proj1 (every_in _ _ _) Hst _ Hev).
  - destruct (eval_simple vars closure) as [w | ] eqn:Hw; [ | discriminate ].
    pose proof (simple_pa_ok _ _ _ _ Hvars Hw) as Hwpa.
    unfold project_function_slot in Hev.
    destruct w; simpl in Hev, Hwpa; try discriminate;
      try (destruct (lookup s (static_values statics)) as [[ ] | ];
           try discriminate);
      repeat (first
                [ match type of Hev with
                  | context [if ?b then _ else _] => destruct b
                  end
                | match type of Hev with
                  | context [lookup ?k ?m] => destruct (lookup k m)
                  end ];
              simpl in Hev; try discriminate);
      injection Hev as <-; first [ exact Hwpa | exact I ].
Qed.

Lemma fields_from_impl (P Q : nat -> value -> Prop) (k : nat)
    (fields : list value) :
  (∀ (i : nat) (field : value), field ∈ fields -> P i field -> Q i field) ->
  fields_from P k fields -> fields_from Q k fields.
Proof.
  revert k. induction fields as [ | field fields IH]; intros k H; simpl;
    [ auto | ].
  intros [Hf Hfs]. split; [ exact (H k field (or_introl eq_refl) Hf) | ].
  apply IH; [ | exact Hfs ]. intros i f Hin. exact (H i f (or_intror Hin)).
Qed.

(* A call kind that calls a closure of the code [id], with arguments of
   the arity [a], as the direct call of [id] does: that direct call, the
   call of unknown arity, or a call of known arity that allows [id] and
   has a group of arguments at least. *)
Definition calls_as_direct (kind : call_kind) (id : code_id) (a : arity)
    : Prop :=
  kind = Indirect_unknown_arity ∨ kind = Direct id
  ∨ (∃ code_ids : option (list code_id),
       kind = Indirect_known_arity code_ids
       ∧ code_ids_allow code_ids id = true)
    ∧ a <> [].

(* Such a call of a closure of code that is not tupled, with as many
   groups of arguments as the code has parameters, is a full call. *)
Lemma shape_of_closure_call_as_direct (statics : static_env)
    (closure : value) (kind : call_kind) (vs : list value)
    (args_arity return_arity : arity) (called : code_id)
    (slots : list (value_slot * value)) (c : code) :
  closure_of statics closure = Some (called, slots) ->
  calls_as_direct kind called args_arity ->
  lookup called (static_code statics) = Some c ->
  code_is_tupled c = false ->
  length args_arity = length (code_params_arity c) ->
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some (Full_call closure c vs args_arity).
Proof.
  intros Hcl Hk Hcode Htup Hfull.
  destruct Hk as [Hk | [Hk | ((code_ids & -> & Hallow) & Hne)]];
    [ exact (shape_of_closure_call_full _ _ _ _ _ _ _ _ _ Hcl (or_introl Hk)
               Hcode Htup Hfull)
    | exact (shape_of_closure_call_full _ _ _ _ _ _ _ _ _ Hcl (or_intror Hk)
               Hcode Htup Hfull) | ].
  assert (Hkf : known_arity_full statics called args_arity = true).
  { unfold known_arity_full. rewrite Hcode.
    destruct args_arity as [ | a0 rest]; [ contradiction | ].
    cbv beta iota. rewrite Htup, Hfull, Nat.eqb_refl. reflexivity. }
  rewrite <- (shape_of_closure_call_full statics closure
                Indirect_unknown_arity vs args_arity return_arity called slots
                c Hcl (or_introl eq_refl) Hcode Htup Hfull).
  unfold shape_of_closure_call. rewrite Hcl. cbv beta iota zeta.
  rewrite Hallow, Hkf. reflexivity.
Qed.

(* Such a call is that call of the closure. *)
Lemma shape_of_call_as_direct (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (called : code_id) (slots : list (value_slot * value)) :
  closure_of statics closure = Some (called, slots) ->
  calls_as_direct kind called args_arity ->
  shape_of_call statics closure kind vs args_arity return_arity
  = shape_of_closure_call statics closure kind vs args_arity return_arity.
Proof.
  intros Hcl [-> | [-> | ((code_ids & -> & _) & _)]]; [ | reflexivity
    | reflexivity ].
  unfold shape_of_call. destruct closure; try discriminate Hcl; reflexivity.
Qed.

(* The run, for any premise [slots_read] of the facts of a projection
   that gives the rule of the fix: that no field is local, before the
   fix, or anything if the solution has the rule. *)
Module WithLocal.

Section Run.

Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable keep : value_slot -> bool.
Variable codes : list (code_id * code).
Variable symbols : list symbol.
Variable u : unit_interface.
(* Whether the primitives of the heap are in the term. *)
Variable heap_prims : Prop.
(* The allocation of each location of the run. *)
Variable located : location -> option node.
(* What the facts of a projection carry ([named_edges]). *)
Variable slots_read : Prop.
Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hconsts : g_any_source G Node_all_constants.
Hypothesis Hsyms :
  ∀ s : symbol, noted s -> may_come_from sol (Node_symbol s) (Node_symbol s).
Hypothesis Hunit_syms :
  ∀ s : symbol,
    among s symbols = true -> may_come_from sol (Node_symbol s) (Node_symbol s).
Hypothesis Hdeps : ∀ d : closure_dep, deps d -> closure_dep_facts G cds d.
Hypothesis Hnd : NoDup (map fst cds).
Hypothesis Hcds :
  ∀ (id : code_id) (cd : code_dep),
    lookup id cds = Some cd -> known_witness_edges G id cd.
Hypothesis Hslots_read : slots_read -> local_reads_used is_local G sol.

Abbreviation slots := slots_read.
Abbreviation called := (called_by G cds).
Abbreviation flows_ := (flows G sol keep called located).
Abbreviation env_flows_ := (env_flows G sol keep called located).
Abbreviation statics_flow_ := (statics_flow G sol keep called located).
Abbreviation heap_flows_ := (heap_flows G sol keep called located).
Abbreviation as_ n := (s_any_source sol n = true).
Abbreviation au n := (s_any_usage sol n = true).
Abbreviation hu n := (s_has_usage sol n = true).

Lemma not_local (f : field) :
  (∀ w : value_slot, f <> Field_value_slot w) ->
  (∀ fs : function_slot, f <> Field_function_slot fs) ->
  is_local f = false.
Proof.
  intros Hw Hf. destruct (is_local f) eqn:E; [ | reflexivity ].
  destruct (Hlocal f E) as [(w & ->) | (fs & ->)];
    [ destruct (Hw w) | destruct (Hf fs) ]; reflexivity.
Qed.

Ltac not_local_field := apply not_local; intros; discriminate.

Lemma consts_any_source : as_ Node_all_constants.
Proof. exact (any_source_input _ _ _ Hc _ Hconsts). Qed.

(* A node aliased to the constants has any source, and a node with any
   source is an allocation of its own. *)
Lemma alias_consts_any_source (a : node) :
  g_alias G a Node_all_constants -> as_ a.
Proof.
  intros Hal.
  exact (any_source_alias _ _ _ Hc _ _ (alias_input _ _ _ Hc _ _ Hal)
           consts_any_source).
Qed.

Lemma any_source_self (a : node) : as_ a -> may_come_from sol a a.
Proof.
  intros Has. split; [ | split; [ | split; [ | split ] ] ].
  - exact (as_hs _ _ _ Hc _ Has).
  - left. exact Has.
  - auto.
  - auto.
  - intros _ u' Hu. left. exact Hu.
Qed.

(* A node with a constructor fact whose field has a source is an
   allocation of its own. *)
Lemma alloc_self (a : node) (f : field) (from : node) :
  g_constructor G a f from -> s_has_source sol from = true ->
  may_come_from sol a a.
Proof.
  intros Hcons Hhs. split; [ | split; [ | split; [ | split ] ] ].
  - exact (has_source_constructor _ _ _ Hc _ _ _ Hhs Hcons).
  - destruct (s_any_source sol a) eqn:E; [ left; reflexivity | right ].
    exact (sources_constructor _ _ _ Hc _ _ _ Hcons E).
  - auto.
  - auto.
  - intros _ u' Hu. left. exact Hu.
Qed.

(* The symbol of a static boxed number is an allocation of its own. *)
Lemma boxed_self (s : symbol) (contents : simple) :
  boxed_edges G s contents -> may_come_from sol (Node_symbol s) (Node_symbol s).
Proof.
  unfold boxed_edges. destruct (boxed_number_kind contents) as [bn | ].
  - intros Hcons.
    exact (alloc_self _ _ _ Hcons (as_hs _ _ _ Hc _ consts_any_source)).
  - intros [Hal _].
    exact (any_source_self _ (alias_consts_any_source _ Hal)).
Qed.

(* What a static boxed number holds, as [statics_flow] says it. *)
Lemma boxed_flows (vars : env) (s : symbol) (contents : simple) (v : value) :
  boxed_edges G s contents -> eval_simple vars contents = Some v ->
  as_ (Node_symbol s)
  ∨ ∀ (c : const) (bn : boxable_number),
      v = Value_const c -> is_number_of_kind bn c = true ->
      g_constructor G (Node_symbol s) (Field_boxed_number bn)
        Node_all_constants.
Proof.
  unfold boxed_edges. intros Hb Hv.
  destruct (boxed_number_kind contents) as [bn | ] eqn:Hk.
  - right. intros c bn' -> Hk'.
    destruct contents as [x | sym | c']; try discriminate Hk.
    simpl in Hv. injection Hv as ->.
    rewrite (boxed_number_kind_of_kind _ _ Hk') in Hk. injection Hk as ->.
    exact Hb.
  - left. exact (alias_consts_any_source _ (proj1 Hb)).
Qed.

(* A node with any source, used only if [a] has any usage, may hold what
   comes from [a]. *)
Lemma may_come_from_any (a n : node) :
  as_ n -> (hu n -> au a) -> may_come_from sol a n.
Proof.
  intros Has Hau. split; [ | split; [ | split; [ | split ] ] ].
  - exact (as_hs _ _ _ Hc _ Has).
  - left. exact Has.
  - intros _. exact Has.
  - intros Hn. exact (Hau (au_hu _ _ _ Hc _ Hn)).
  - intros Hn u' _. right. exact (Hau Hn).
Qed.

(* The allocation of a closure is one of its own: its call witness has a
   source. *)
Lemma called_alloc (b : node) (id : code_id) :
  called b id -> may_come_from sol b b.
Proof.
  unfold called_by, closure_dep_facts. simpl.
  destruct (lookup id cds) as [cd | ] eqn:Hl.
  - intros (_ & Hcons & _). apply (alloc_self _ _ _ Hcons).
    destruct (Hcds id cd Hl) as (_ & _ & _ & Hcode & Has).
    exact (has_source_constructor _ _ _ Hc _ _ _
             (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Has)) Hcode).
  - intros (w & Has & Hcons). apply (alloc_self _ _ _ Hcons).
    exact (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Has)).
Qed.

Lemma simple_flows (vars : env) (s : simple) (v : value) :
  env_flows_ vars -> simple_noted noted s ->
  eval_simple vars s = Some v -> flows_ v (node_of_simple s).
Proof.
  intros Hvars Hnoted Hs. destruct s as [x | sym | c]; simpl in Hs, Hnoted |- *.
  - exact (Hvars x v Hs).
  - injection Hs as <-. exact (Hsyms sym Hnoted).
  - injection Hs as <-. exact consts_any_source.
Qed.

(* The value of a variable is at its node, and so is no location that
   has no allocation: none is a value of the arguments of a primitive. *)
Lemma env_flows_no_location (vars : env) (args : list simple)
    (vs : list value) (l : location) :
  located l = None -> env_flows_ vars -> eval_simples vars args = Some vs ->
  Value_location l ∉ vs.
Proof.
  intros Hnone Hvars Hvs Hin.
  destruct (eval_simples_in _ _ _ _ Hvs Hin) as (s & _ & Hs).
  destruct s as [y | sym | c]; simpl in Hs.
  - pose proof (Hvars y _ Hs) as H. simpl in H. rewrite Hnone in H.
    exact H.
  - discriminate Hs.
  - discriminate Hs.
Qed.

(* A [Let] of a term without the primitives of the heap is not of one. *)
Lemma named_edges_not_heap (sr : Prop) (ctx : option code_id) (x : variable)
    (d : named) :
  named_edges G noted deps keep sr False ctx x d ->
  ∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args.
Proof.
  intros H q args ->. simpl in H. destruct H as [[[Hoff | []] _] _].
  discriminate Hoff.
Qed.

Lemma named_edges_no_heap (sr : Prop) (ctx : option code_id) (x : variable)
    (d : named) :
  ~ heap_prims -> named_edges G noted deps keep sr heap_prims ctx x d ->
  ∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args.
Proof.
  intros Hnh H q args ->. simpl in H.
  destruct H as [[[Hoff | Hh] _] _]; [ discriminate Hoff | exact (Hnh Hh) ].
Qed.

(* What is said of a block at an allocation. *)
Definition block_structure (a : node) (fields : list value) : Prop :=
  g_constructor G a Field_is_int Node_all_constants
  ∧ g_constructor G a Field_get_tag Node_all_constants
  ∧ fields_from
      (fun i field =>
         ∃ from : node,
           g_constructor G a (Field_block i Value) from
           ∧ flows_ field from)
      0 fields.

Lemma flows_block (tag : Z) (fields : list value) (n : node) :
  flows_ (Value_block tag fields) n
  <-> ∃ a : node, may_come_from sol a n ∧ block_structure a fields.
Proof. split; intros H; exact H. Qed.

Lemma block_structure_made (a : node) (fields : list simple) (vars : env)
    (vs : list value) :
  block_edges G noted a fields -> env_flows_ vars ->
  eval_simples vars fields = Some vs -> block_structure a vs.
Proof.
  intros (Hcons & His & Hgt & Hnoted) Hvars Hvs.
  split; [ exact His | split; [ exact Hgt | ] ].
  apply fields_from_intro. intros i v Hv.
  destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
  exists (node_of_simple s). split; [ exact (Hcons i s Hs) | ].
  exact (simple_flows _ _ _ Hvars (Hnoted s (nth_error_In _ _ Hs)) Hsv).
Qed.

(* A block read through a symbol is the static block it is bound to. *)
Lemma block_of_flows (statics : static_env) (v : value) (n : node) (tag : Z)
    (fields : list value) :
  statics_flow_ statics -> flows_ v n ->
  block_of statics v = Some (tag, fields) ->
  ∃ a : node, may_come_from sol a n ∧ block_structure a fields.
Proof.
  intros Hst Hv Hb.
  destruct (FlambdaPrimitiveProofs.block_of_cases _ _ _ _ Hb)
    as [-> | (s & -> & Hl)].
  - exact Hv.
  - exists (Node_symbol s). split; [ exact Hv | ].
    exact (proj2 (Hst s _ Hl)).
Qed.

Lemma boxed_number_of_cases (statics : static_env) (v : value) (c : const) :
  boxed_number_of statics v = Some c ->
  v = Value_boxed_number c
  ∨ ∃ s : symbol,
      v = Value_symbol s
      ∧ lookup s (static_values statics)
        = Some (Static_boxed_number (Value_const c)).
Proof.
  destruct v as [c' | | | s | | | c' | | ]; simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [[ | | | contents | | | ] | ]
      eqn:Hl; try discriminate.
    destruct contents as [c' | | | | | | | | ]; try discriminate.
    unfold boxable_contents. destruct (boxable_number_of_const c');
      [ | discriminate ].
    intros [= <-]. right. exists s. auto.
  - unfold boxable_contents. destruct (boxable_number_of_const c');
      [ | discriminate ].
    intros [= <-]. left. reflexivity.
Qed.

Lemma is_tagged_immediate_const (v : value) :
  is_tagged_immediate v = true -> ∃ c : const, v = Value_const c.
Proof. destruct v; try discriminate. intros _. eexists. reflexivity. Qed.

(* A read of a block at [base], of a field that is a constant. *)
Lemma read_const (a base : node) (x : variable) (f : field) (c : const) :
  is_local f = false ->
  g_constructor G a f Node_all_constants ->
  g_accessor G (Node_variable x) f base ->
  may_come_from sol a base ->
  as_ (Node_variable x).
Proof.
  intros Hl Hcons Hacc Ha.
  exact (flows_read _ _ _ keep called located Hc _ _ _ _ _ (Value_const c) Hl
           Hcons
    Hacc Ha
           consts_any_source).
Qed.

(* ------------------------------------------------------------------------ *)
(* Primitives                                                               *)
(* ------------------------------------------------------------------------ *)

(* The value of a primitive that the traversal does not read in a way of
   its own is at its variable, which has any source and a use of each
   argument: the value is a constant, or an argument or a part of one,
   which the rules give any usage when the variable has a usage. *)
Lemma default_flows (statics : static_env) (vars : env) (x : variable)
    (p : prim) (args : list simple) (vs : list value) (v : value) :
  statics_flow_ statics -> env_flows_ vars ->
  default_prim_edges G x args ->
  prim_classified p args = false ->
  eval_simples vars args = Some vs ->
  eval_prim statics p vs = Some v ->
  flows_ v (Node_variable x).
Proof.
  intros Hst Hvars [Has Huse] Hcl Hvs Hev.
  pose proof (any_source_input _ _ _ Hc _ Has) as Hasx.
  (* Each argument is a constant, or is used by [x]. *)
  assert (Harg : ∀ (s : simple) (w : value),
    s ∈ args -> eval_simple vars s = Some w ->
    (∃ c : const, w = Value_const c)
    ∨ (∃ y : variable,
         flows_ w (Node_variable y)
         ∧ (hu (Node_variable x) -> au (Node_variable y)))
    ∨ (∃ sym : symbol,
         w = Value_symbol sym
         ∧ (hu (Node_variable x) -> au (Node_symbol sym)))).
  { intros s w Hin Hs. destruct s as [y | sym | c]; simpl in Hs.
    - right. left. exists y. split; [ exact (Hvars y w Hs) | ].
      intros Hhu.
      exact (any_usage_use _ _ _ Hc _ _ Hhu (Huse _ (free_names_var _ _ Hin))).
    - injection Hs as <-. right. right. exists sym. split; [ reflexivity | ].
      intros Hhu.
      exact (any_usage_use _ _ _ Hc _ _ Hhu
               (Huse _ (free_names_symbol _ _ Hin))).
    - injection Hs as <-. left. eexists. reflexivity. }
  destruct (default_prim_result statics p args vs v Hcl
              (eval_simples_length _ _ _ Hvs) Hev)
    as [(c & ->) | [-> | [(arr & idx & elements & -> & Harr & Hin)
                         | (-> & Hbound)]]].
  - exact Hasx.
  - (* The identity. *)
    destruct (eval_simples_in _ _ _ _ Hvs (or_introl eq_refl))
      as (s & Hin & Hs).
    destruct (Harg s v Hin Hs)
      as [(c & ->) | [(y & Hy & Hu) | (sym & -> & Hu)]].
    + exact Hasx.
    + exact (flows_escape _ _ _ _ _ _ Hc _ _ _ Hasx Hu Hy).
    + exact (may_come_from_any _ _ Hasx Hu).
  - (* A load from an immutable array. *)
    destruct (eval_simples_in _ _ _ _ Hvs (or_introl eq_refl))
      as (s & Hins & Hs).
    destruct (array_of_cases _ _ _ Harr) as [-> | (sym & -> & Hl)].
    + destruct (Harg s _ Hins Hs)
        as [(c & Hc') | [(y & Hy & Hu) | (sym & Hsym & _)]];
        try discriminate.
      destruct Hy as [_ Hall].
      destruct (proj1 (every_in _ _ _) Hall v Hin)
        as [(c & ->) | (m & Hm & Hmu)]; [ exact Hasx | ].
      apply (flows_escape _ _ _ _ _ _ Hc v _ m Hasx); [ | exact Hm ].
      intros Hhu. apply Hmu, (au_hu _ _ _ Hc), Hu, Hhu.
    + (* A static array: its symbol has any usage if [x] has a use. *)
      assert (Hau : hu (Node_variable x) -> au (Node_symbol sym)).
      { destruct (Harg s _ Hins Hs)
          as [(c & Hc') | [(y & Hy & Hu) | (sym' & Hsym & Hu)]];
          [ discriminate Hc' | | ].
        - intros Hhu. exact (proj1 (proj2 (proj2 (proj2 Hy))) (Hu Hhu)).
        - injection Hsym as <-. exact Hu. }
      destruct (In_nth_error _ _ Hin) as (i & Hi).
      destruct (fields_from_elim _ 0 _ i v (proj2 (Hst sym _ Hl)) Hi)
        as (from & Hcons & Hfrom).
      apply (flows_escape _ _ _ _ _ _ Hc v _ from Hasx); [ | exact Hfrom ].
      intros Hhu.
      exact (any_usage_constructor _ _ _ Hc _ _ _ (Hau Hhu) Hcons
               ltac:(not_local_field)).
  - (* An immutable array made. *)
    split; [ exact Hasx | ]. apply (every_in _ _ _). intros w Hw.
    destruct (eval_simples_in _ _ _ _ Hvs Hw) as (s & Hin & Hs).
    destruct (Harg s w Hin Hs) as [Hconst | [(y & Hy & Hu) | (sym & -> & Hu)]].
    + left. exact Hconst.
    + right. exists (Node_variable y). auto.
    + right. exists (Node_symbol sym). split; [ | exact Hu ].
      specialize (Hbound _ Hw). simpl in Hbound.
      destruct (lookup sym (static_values statics)) as [allocated | ] eqn:Hl;
        [ | discriminate ].
      exact (proj1 (Hst sym allocated Hl)).
Qed.

(* The value of a [Let] is at the node of its variable. *)
Lemma named_flows (statics : static_env) (vars : env) (ctx : option code_id)
    (x : variable) (d : named) (v : value) :
  statics_flow_ statics -> env_flows_ vars ->
  named_edges G noted deps keep slots heap_prims ctx x d ->
  eval_named statics vars d = Some v ->
  flows_ v (Node_variable x).
Proof.
  intros Hst Hvars Hedges Hev.
  destruct d as [s | p args | | slot decls vs | | from w closure
               | move_from move_to closure];
    simpl in Hedges; try contradiction; simpl in Hev.
  - destruct Hedges as [Hal Hnoted].
    apply (flows_alias _ _ _ _ _ _ Hc v _ _ (alias_input _ _ _ Hc _ _ Hal)).
    exact (simple_flows _ _ _ Hvars Hnoted Hev).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    unfold prim_edges in Hedges. destruct Hedges as [_ Hedges].
    destruct (prim_classified p args) eqn:Hcl;
      [ | exact (default_flows _ _ _ _ _ _ _ Hst Hvars Hedges Hcl Hvs Hev) ].
    destruct p as [p' | b | p' | h | r].
    + destruct p' as [kind mut fld | [ | ] | | | ak | | ok | ik op | src dst
                     | | bn | bn | | | r];
        destruct args as [ | s1 [ | s2 args]]; simpl in Hcl;
        try discriminate Hcl; simpl in Hedges;
        destruct Hedges as [Hedge Hnoted]; simpl in Hvs;
        destruct (eval_simple vars s1) as [v1 | ] eqn:Hs1; try discriminate Hvs;
        injection Hvs as <-; rewrite eval_prim_unary in Hev;
        cbn [eval_unary_primitive] in Hev;
        pose proof (simple_flows _ _ _ Hvars Hnoted Hs1) as Hv1.
      * (* [Block_load] *)
        destruct kind as [tag size fk]. unfold eval_block_load in Hev.
        destruct (block_of statics v1) as [[btag fields] | ] eqn:Hb;
          [ | discriminate Hev ].
        destruct (known_is tag btag && known_is size (Z.of_nat (length fields))
                  && (0 <=? fld)%Z); [ | discriminate Hev ].
        destruct (nth_error fields (Z.to_nat fld)) as [loaded | ] eqn:Hnth;
          [ | discriminate Hev ].
        destruct (match fk with
                  | Any_value => kind_eqb (kind_of_value loaded) Value
                  | Immediate => is_tagged_immediate loaded
                  end); [ | discriminate Hev ].
        injection Hev as <-.
        destruct (block_of_flows _ _ _ _ _ Hst Hv1 Hb)
          as (a & Ha & _ & _ & Hfields).
        destruct (fields_from_elim _ _ _ _ _ Hfields Hnth)
          as (from & Hcons & Hfrom).
        refine (flows_read _ _ _ _ _ _ Hc _ _ _ _ _ _ _ Hcons Hedge Ha Hfrom).
        not_local_field.
      * (* [Is_int] *)
        unfold option_map, eval_is_int in Hev.
        destruct (is_tagged_immediate v1) eqn:Ht.
        -- injection Hev as <-. simpl.
           destruct (is_tagged_immediate_const _ Ht) as (c & ->).
           apply (any_source_accessor _ _ _ Hc _ _ _ Hv1 Hedge).
           not_local_field.
        -- destruct (is_some (block_of statics v1)) eqn:Hb;
             [ | discriminate Hev ].
           injection Hev as <-. simpl.
           destruct (block_of statics v1) as [[tag fields] | ] eqn:Hb';
             [ | discriminate Hb ].
           destruct (block_of_flows _ _ _ _ _ Hst Hv1 Hb')
             as (a & Ha & His & _ & _).
           refine (read_const a _ x Field_is_int (Naked_immediate 0) _ His
                     Hedge Ha).
           not_local_field.
      * (* [Get_tag] *)
        destruct (block_of statics v1) as [[tag fields] | ] eqn:Hb;
          [ | discriminate Hev ].
        injection Hev as <-. simpl.
        destruct (block_of_flows _ _ _ _ _ Hst Hv1 Hb)
          as (a & Ha & _ & Hgt & _).
        refine (read_const a _ x Field_get_tag (Naked_immediate tag) _ Hgt
                  Hedge Ha).
        not_local_field.
      * (* [Unbox_number] *)
        destruct (boxed_number_of statics v1) as [c | ] eqn:Hbx;
          [ | discriminate Hev ].
        destruct (is_number_of_kind bn c) eqn:Hk; [ | discriminate Hev ].
        injection Hev as <-.
        destruct (boxed_number_of_cases _ _ _ Hbx) as [-> | (s & -> & Hl)].
        -- destruct Hv1 as (a & bn' & from & Ha & Hk' & Hcons & Hfrom).
           rewrite (is_number_of_kind_unique _ _ _ Hk' Hk) in Hcons.
           refine (flows_read _ _ _ _ _ _ Hc _ _ _ _ _ (Value_const c) _ Hcons
             Hedge
                     Ha Hfrom).
           not_local_field.
        -- destruct (Hst s _ Hl) as [_ [Has | Hcons]].
           ++ exact (any_source_accessor _ _ _ Hc _ _ _
                       (proj1 (proj2 (proj2 Hv1)) Has) Hedge
                       ltac:(not_local_field)).
           ++ refine (flows_read _ _ _ _ _ _ Hc _ _ _ _ _ (Value_const c) _
                        (Hcons c bn eq_refl Hk) Hedge Hv1 consts_any_source).
              not_local_field.
      * (* [Box_number] *)
        destruct v1 as [c | | | | | | | | ]; try discriminate Hev.
        destruct (is_number_of_kind bn c) eqn:Hk; [ | discriminate Hev ].
        injection Hev as <-. simpl.
        exists (Node_variable x), bn, (node_of_simple s1).
        split; [ exact (alloc_self _ _ _ Hedge (as_hs _ _ _ Hc _ Hv1)) | ].
        auto.
    + discriminate Hcl.
    + destruct p' as [[tag] | ak]; [ | discriminate Hcl ].
      rewrite eval_prim_variadic in Hev. cbn [eval_variadic_primitive] in Hev.
      destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs
                && forallb (bound_if_symbol statics) vs);
        [ | discriminate Hev ].
      injection Hev as <-. apply flows_block.
      exists (Node_variable x). destruct Hedges as (Hcons & His & Hgt & Hnoted).
      split.
      * exact (alloc_self _ _ _ His (as_hs _ _ _ Hc _ consts_any_source)).
      * exact (block_structure_made _ _ _ _
                 (conj Hcons (conj His (conj Hgt Hnoted))) Hvars Hvs).
    + destruct (eval_prim_arity _ _ _ _ Hev)
        as [(p' & v1 & Hp & _) | [(p' & v1 & v2 & Hp & _) | (p' & Hp & _)]];
        discriminate Hp.
    + destruct (eval_prim_arity _ _ _ _ Hev)
        as [(p' & v1 & Hp & _) | [(p' & v1 & v2 & Hp & _) | (p' & Hp & _)]];
        discriminate Hp.
  - (* [Closure_of_set] *)
    destruct (lookup slot decls) as [id0 | ] eqn:Hslot; [ | discriminate Hev ].
    destruct (eval_value_slots vars vs) as [vvs | ] eqn:Hvvs;
      [ | discriminate Hev ].
    injection Hev as <-.
    destruct (Hedges id0 Hslot) as (sites & Hx & Hall).
    assert (Hclo : ∀ (f : function_slot) (id : code_id),
      lookup f decls = Some id ->
      may_come_from sol (sites f) (sites f)
      ∧ called (sites f) id
      ∧ (∀ (g : function_slot) (id' : code_id),
           lookup g decls = Some id' ->
           g_constructor G (sites f) (Field_function_slot g) (sites g))
      ∧ for_all_bindings
          (fun w field =>
             keep w = true ->
             ∃ from : node,
               g_constructor G (sites f) (Field_value_slot w) from
               ∧ flows_ field from)
          vvs).
    { intros f id Hf. destruct (Hall f id Hf) as (Hdep & Hfs & Hvs).
      pose proof (Hdeps _ Hdep) as Hcalled.
      split; [ exact (called_alloc _ _ Hcalled) | ].
      split; [ exact Hcalled | split; [ exact Hfs | ] ].
      apply for_all_bindings_lookup. intros w' v' Hl Hk.
      destruct (eval_value_slots_lookup _ _ _ _ _ Hvvs Hl) as (s & Hs & Hsv).
      destruct (Hvs w' s Hs Hk) as [Hcons Hnoted].
      exists (node_of_simple s).
      split; [ exact Hcons | exact (simple_flows _ _ _ Hvars Hnoted Hsv) ]. }
    simpl. exists sites. split; [ | exact Hclo ].
    rewrite <- Hx. exact (proj1 (Hclo slot id0 Hslot)).
  - (* [Project_value_slot_from] *)
    destruct Hedges as ((Hacc & Hnoted) & Hk & Hslots).
    destruct (eval_simple vars closure) as [vc | ] eqn:Hvc;
      [ | discriminate Hev ].
    pose proof (simple_flows _ _ _ Hvars Hnoted Hvc) as Hfc.
    apply project_value_slot_defined in Hev as (decls & slots & Hcis & Hw).
    destruct vc as [ | code slots0 | slot0 decls0 slots0 | s | | | | | ];
      simpl in Hcis; try discriminate Hcis.
    + destruct Hfc.
    + destruct (lookup slot0 decls0) as [id0 | ] eqn:Hl0;
        [ | discriminate Hcis ].
      injection Hcis as -> <- <-.
      destruct Hfc as (sites & Hn & Hall).
      destruct (Hall from id0 Hl0) as (_ & _ & _ & Hvs).
      destruct (proj1 (for_all_bindings_lookup _ _ _) Hvs w v Hw Hk)
        as (from' & Hcons & Hfrom).
      exact (flows_read_slot _ _ _ _ _ _ Hc _ _ _ _ _ v (Hslots_read Hslots)
               Hcons Hacc Hn Hfrom).
    + destruct (lookup s (static_values statics))
        as [[ | slot0 syms0 decls0 slots0 | | | | | ] | ] eqn:Hls;
        try discriminate Hcis.
      * destruct (Hst s _ Hls) as [_ []].
      * destruct (lookup slot0 decls0) as [id0 | ] eqn:Hl0;
          [ | discriminate Hcis ].
        injection Hcis as -> <- <-.
        destruct (Hst s _ Hls) as (_ & _ & _ & Hvs).
        destruct (proj1 (for_all_bindings_lookup _ _ _) Hvs w v Hw Hk)
          as (from' & Hcons & Hfrom).
        exact (flows_read_slot _ _ _ _ _ _ Hc _ _ _ _ _ v
                 (Hslots_read Hslots) Hcons Hacc Hfc Hfrom).
  - (* [Project_function_slot] *)
    destruct Hedges as ((Hacc & Hnoted) & Hslots).
    destruct (eval_simple vars closure) as [vc | ] eqn:Hvc;
      [ | discriminate Hev ].
    pose proof (simple_flows _ _ _ Hvars Hnoted Hvc) as Hfc.
    unfold project_function_slot in Hev.
    destruct vc as [ | code slots0 | slot0 decls0 slots0 | s | | | | | ];
      simpl in Hev; try discriminate Hev.
    + destruct Hfc.
    + destruct (Pos.eqb_spec slot0 move_from) as [-> | _];
        [ | discriminate Hev ].
      destruct (lookup move_from decls0) as [id0 | ] eqn:Hl0;
        [ | discriminate Hev ].
      destruct (lookup move_to decls0) as [id1 | ] eqn:Hl1;
        [ | discriminate Hev ].
      injection Hev as <-.
      destruct Hfc as (sites & Hn & Hall).
      destruct (Hall move_from id0 Hl0) as (_ & _ & Hfs & _).
      apply (flows_read_slot _ _ _ _ _ _ Hc _ _ _ _ _ _ (Hslots_read Hslots)
               (Hfs move_to id1 Hl1) Hacc Hn).
      simpl. exists sites.
      split; [ exact (proj1 (Hall move_to id1 Hl1)) | exact Hall ].
    + destruct (lookup s (static_values statics))
        as [[ | slot0 syms0 decls0 slots0 | | | | | ] | ] eqn:Hls;
        try discriminate Hev.
      * destruct (Hst s _ Hls) as [_ []].
      * destruct (Pos.eqb_spec slot0 move_from) as [-> | _];
          [ | discriminate Hev ].
        destruct (lookup move_from decls0) as [id0 | ];
          [ | discriminate Hev ].
        destruct (lookup move_to decls0) as [id1 | ]; [ | discriminate Hev ].
        destruct (lookup move_to syms0) as [s' | ] eqn:Hs';
          [ | discriminate Hev ].
        injection Hev as <-.
        destruct (Hst s _ Hls) as (_ & _ & Hfs & _).
        destruct (Hfs move_to s' Hs') as [Hcons Hself].
        exact (flows_read_slot _ _ _ _ _ _ Hc _ _ _ _ _ (Value_symbol s')
                 (Hslots_read Hslots) Hcons Hacc Hfc Hself).
Qed.

(* ------------------------------------------------------------------------ *)
(* Static bindings                                                          *)
(* ------------------------------------------------------------------------ *)

Abbreviation edges :=
  (expr_edges G noted deps cds keep codes symbols slots heap_prims).
Abbreviation handler_edges_ :=
  (handler_edges G noted deps cds keep codes symbols slots heap_prims).
Abbreviation binding_edges_ :=
  (binding_edges G noted deps cds keep codes symbols slots heap_prims).

(* The code bound so far is the code of the unit, as the graph has it. *)
Definition codes_ok (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c -> binding_edges_ (Bind_code id c).

(* A [Let_static] of the fragment binds symbols and code as the graph
   says. *)
Lemma statics_bindings (vars : env) (bindings : list static_binding)
    (statics statics' : static_env) :
  env_flows_ vars -> every binding_edges_ bindings ->
  statics_flow_ statics -> codes_ok statics ->
  eval_static_bindings vars bindings statics = Some statics' ->
  statics_flow_ statics' ∧ codes_ok statics'.
Proof.
  intros Hvars. revert statics.
  induction bindings as [ | b bindings IH];
    intros statics Hedges Hst Hcodes Heval; simpl in Heval.
  - injection Heval as <-. split; [ exact Hst | exact Hcodes ].
  - destruct Hedges as [Hb Hbindings].
    destruct b as [id c | id | s const].
    + destruct (lookup id (static_code statics)); [ discriminate Heval | ].
      refine (IH _ Hbindings _ _ Heval).
      * exact Hst.
      * intros id' c' Hl. simpl in Hl.
        destruct (Pos.eqb_spec id' id) as [-> | _];
          [ injection Hl as <-; exact Hb | exact (Hcodes id' c' Hl) ].
    + exact (IH _ Hbindings Hst Hcodes Heval).
    + destruct (lookup s (static_values statics)); [ discriminate Heval | ].
      destruct const as [ | slot syms decls vs | t fields | contents | bytes
                         | elements ];
        simpl in Hb; try contradiction; simpl in Heval.
      * (* A static closure of a set. *)
        destruct (lookup slot decls) as [id0 | ] eqn:Hslot;
          [ | discriminate Heval ].
        destruct (eval_value_slots vars vs) as [vvs | ] eqn:Hvvs;
          [ | discriminate Heval ].
        refine (IH _ Hbindings _ _ Heval); [ | exact Hcodes ].
        intros s' allocated Hl. simpl in Hl.
        destruct (Pos.eqb_spec s' s) as [-> | _];
          [ | exact (Hst s' allocated Hl) ].
        injection Hl as <-.
        destruct Hb as (_ & Hdep & Hfs & Hvs).
        pose proof (Hdeps _ (Hdep id0 Hslot)) as Hcalled.
        split; [ exact (called_alloc _ _ Hcalled) | ].
        split; [ intros id Hid; rewrite Hslot in Hid; injection Hid as <-;
                 exact Hcalled | ].
        split.
        -- intros g s' Hg. destruct (Hfs g s' Hg) as [Hcons Hamong].
           split; [ exact Hcons | exact (Hunit_syms s' Hamong) ].
        -- apply for_all_bindings_lookup. intros w v Hl Hk.
           destruct (eval_value_slots_lookup _ _ _ _ _ Hvvs Hl)
             as (x & Hx & Hxv).
           destruct (Hvs w x Hx Hk) as [Hcons Hnoted].
           exists (node_of_simple x).
           split; [ exact Hcons | exact (simple_flows _ _ _ Hvars Hnoted Hxv) ].
      * (* A static block. *)
        destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
          [ | discriminate Heval ].
        refine (IH _ Hbindings _ _ Heval); [ | exact Hcodes ].
        intros s' allocated Hl. simpl in Hl.
        destruct (Pos.eqb_spec s' s) as [-> | _];
          [ | exact (Hst s' allocated Hl) ].
        injection Hl as <-. destruct Hb as (Hcons & His & Hgt & Hnoted).
        split.
        -- exact (alloc_self _ _ _ His (as_hs _ _ _ Hc _ consts_any_source)).
        -- exact (block_structure_made _ _ _ _
                    (conj Hcons (conj His (conj Hgt Hnoted))) Hvars Hvs).
      * (* A static boxed number. *)
        destruct (eval_simple vars contents) as [v | ] eqn:Hv;
          [ | discriminate Heval ].
        refine (IH _ Hbindings _ _ Heval); [ | exact Hcodes ].
        intros s' allocated Hl. simpl in Hl.
        destruct (Pos.eqb_spec s' s) as [-> | _];
          [ | exact (Hst s' allocated Hl) ].
        injection Hl as <-. split; [ exact (boxed_self _ _ Hb) | ].
        exact (boxed_flows _ _ _ _ Hb Hv).
      * (* A static string. *)
        refine (IH _ Hbindings _ _ Heval); [ | exact Hcodes ].
        intros s' allocated Hl. simpl in Hl.
        destruct (Pos.eqb_spec s' s) as [-> | _];
          [ | exact (Hst s' allocated Hl) ].
        injection Hl as <-. pose proof (alias_consts_any_source _ Hb) as Has.
        split; [ exact (any_source_self _ Has) | exact Has ].
      * (* A static immutable array. *)
        destruct (eval_simples vars elements) as [vs | ] eqn:Hvs;
          [ | discriminate Heval ].
        refine (IH _ Hbindings _ _ Heval); [ | exact Hcodes ].
        intros s' allocated Hl. simpl in Hl.
        destruct (Pos.eqb_spec s' s) as [-> | _];
          [ | exact (Hst s' allocated Hl) ].
        injection Hl as <-.
        destruct elements as [ | element elements ].
        -- (* The empty array, aliased to the constants. *)
           simpl in Hvs. injection Hvs as <-.
           pose proof (alias_consts_any_source _ Hb) as Has.
           split; [ exact (any_source_self _ Has) | exact I ].
        -- destruct Hb as (Hcons & His & Hgt & Hnoted).
           split.
           ++ exact (alloc_self _ _ _ His
                       (as_hs _ _ _ Hc _ consts_any_source)).
           ++ exact (proj2 (proj2 (block_structure_made _ _ _ _
                       (conj Hcons (conj His (conj Hgt Hnoted))) Hvars Hvs))).
Qed.

(* The values that a [Let_static] of the fragment binds keep partial
   applications as the run does. *)
Lemma statics_bindings_pa (vars : env) (bindings : list static_binding)
    (statics statics' : static_env) :
  every binding_edges_ bindings ->
  env_pa_ok statics vars -> statics_pa_ok statics ->
  eval_static_bindings vars bindings statics = Some statics' ->
  statics_pa_ok statics'.
Proof.
  revert statics.
  induction bindings as [ | b bindings IH];
    intros statics Hedges Hvars Hst Heval; simpl in Heval.
  - injection Heval as <-. exact Hst.
  - destruct Hedges as [Hb Hbindings].
    assert (Hstep : ∀ statics1 : static_env,
              statics_le statics statics1 ->
              (∀ (s : symbol) (a : statically_allocated_value),
                 lookup s (static_values statics1) = Some a ->
                 lookup s (static_values statics) = Some a
                 ∨ static_pa_ok statics1 a) ->
              eval_static_bindings vars bindings statics1 = Some statics' ->
              statics_pa_ok statics').
    { intros statics1 Hle Hnew Heval1.
      apply (IH statics1 Hbindings); [ exact (env_pa_ok_le _ _ _ Hle Hvars)
                                     | | exact Heval1 ].
      intros s a Hs. destruct (Hnew s a Hs) as [Hold | Hp]; [ | exact Hp ].
      exact (static_pa_ok_le _ _ _ Hle (Hst s a Hold)). }
    destruct b as [id c | id | s const].
    + destruct (lookup id (static_code statics)) eqn:Hl;
        [ discriminate Heval | ].
      apply (Hstep (Static_env ((id, c) :: static_code statics)
                      (static_values statics))); [ | | exact Heval ].
      * split; [ intros s a H; exact H | ].
        intros id' c' H. simpl.
        destruct (Pos.eqb_spec id' id) as [-> | _]; [ congruence | exact H ].
      * intros s a H. left. exact H.
    + exact (IH statics Hbindings Hvars Hst Heval).
    + destruct (lookup s (static_values statics)) eqn:Hl;
        [ discriminate Heval | ].
      destruct const as [ | slot syms decls vs | t fields | contents | bytes
                         | elements ];
        simpl in Hb; try contradiction; simpl in Heval.
      * destruct (lookup slot decls) as [id0 | ]; [ | discriminate Heval ].
        destruct (eval_value_slots vars vs) as [vvs | ] eqn:Hvvs;
          [ | discriminate Heval ].
        match type of Heval with
        | eval_static_bindings _ _ ?statics1 = _ =>
          assert (Hle : statics_le statics statics1)
        end.
        { split; [ | intros id c H; exact H ].
          intros s' a H. simpl.
          destruct (Pos.eqb_spec s' s) as [-> | _]; [ congruence | exact H ]. }
        apply (Hstep _ Hle); [ | exact Heval ].
        intros s' a H. simpl in H.
        destruct (Pos.eqb_spec s' s) as [-> | _]; [ | left; exact H ].
        injection H as <-. right. simpl.
        exact (value_slots_pa_ok _ _ _ _ (env_pa_ok_le _ _ _ Hle Hvars) Hvvs).
      * destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
          [ | discriminate Heval ].
        match type of Heval with
        | eval_static_bindings _ _ ?statics1 = _ =>
          assert (Hle : statics_le statics statics1)
        end.
        { split; [ | intros id c H; exact H ].
          intros s' a H. simpl.
          destruct (Pos.eqb_spec s' s) as [-> | _]; [ congruence | exact H ]. }
        apply (Hstep _ Hle); [ | exact Heval ].
        intros s' a H. simpl in H.
        destruct (Pos.eqb_spec s' s) as [-> | _]; [ | left; exact H ].
        injection H as <-. right. simpl.
        exact (simples_pa_ok _ _ _ _ (env_pa_ok_le _ _ _ Hle Hvars) Hvs).
      * destruct (eval_simple vars contents) as [v | ] eqn:Hv;
          [ | discriminate Heval ].
        match type of Heval with
        | eval_static_bindings _ _ ?statics1 = _ =>
          assert (Hle : statics_le statics statics1)
        end.
        { split; [ | intros id c H; exact H ].
          intros s' a H. simpl.
          destruct (Pos.eqb_spec s' s) as [-> | _]; [ congruence | exact H ]. }
        apply (Hstep _ Hle); [ | exact Heval ].
        intros s' a H. simpl in H.
        destruct (Pos.eqb_spec s' s) as [-> | _]; [ | left; exact H ].
        injection H as <-. right. simpl.
        exact (simple_pa_ok _ _ _ _ (env_pa_ok_le _ _ _ Hle Hvars) Hv).
      * match type of Heval with
        | eval_static_bindings _ _ ?statics1 = _ =>
          assert (Hle : statics_le statics statics1)
        end.
        { split; [ | intros id c H; exact H ].
          intros s' a H. simpl.
          destruct (Pos.eqb_spec s' s) as [-> | _]; [ congruence | exact H ]. }
        apply (Hstep _ Hle); [ | exact Heval ].
        intros s' a H. simpl in H.
        destruct (Pos.eqb_spec s' s) as [-> | _]; [ | left; exact H ].
        injection H as <-. right. exact I.
      * destruct (eval_simples vars elements) as [vs | ] eqn:Hvs;
          [ | discriminate Heval ].
        match type of Heval with
        | eval_static_bindings _ _ ?statics1 = _ =>
          assert (Hle : statics_le statics statics1)
        end.
        { split; [ | intros id c H; exact H ].
          intros s' a H. simpl.
          destruct (Pos.eqb_spec s' s) as [-> | _]; [ congruence | exact H ]. }
        apply (Hstep _ Hle); [ | exact Heval ].
        intros s' a H. simpl in H.
        destruct (Pos.eqb_spec s' s) as [-> | _]; [ | left; exact H ].
        injection H as <-. right. simpl.
        exact (simples_pa_ok _ _ _ _ (env_pa_ok_le _ _ _ Hle Hvars) Hvs).
Qed.

(* ------------------------------------------------------------------------ *)
(* Continuations                                                            *)
(* ------------------------------------------------------------------------ *)

(* The numbers of the parameters of the continuations in scope. *)
Definition counts (ce : finite_map (list node)) : finite_map nat :=
  map (fun '(k, ns) => (k, length ns)) ce.

(* The code being run is used. *)
Definition live (ctx : option code_id) : Prop :=
  match ctx with
  | None => True
  | Some id => au (Node_code_id id)
  end.

(* In code, the continuations are given values as counted. *)
Definition counted (ctx : option code_id) (ce : finite_map (list node))
    (e : expr) : Prop :=
  ctx <> None -> conts_counted (counts ce) e = true.

Definition handler_counted (ctx : option code_id)
    (ce : finite_map (list node)) (h : continuation_handler) : Prop :=
  ctx <> None -> handler_conts_counted (counts ce) h = true.

(* What is at a node of [ns] is at the node of [targets] at its
   position. *)
Definition transfers (ns targets : list node) : Prop :=
  ∀ (i : nat) (v : value) (n p : node),
    nth_error ns i = Some n -> nth_error targets i = Some p ->
    flows_ v n -> flows_ v p.

(* The number of values that a handler or a way out takes. *)
Definition target_length (c : cont_value) : option nat :=
  match c with
  | Closure h _ _ _ => Some (length (handler_params h))
  | Outer k => Some (length (unit_cont_nodes u k))
  | Extra_args _ _ | Over_application _ _ _ _ _ => None
  end.

(* A continuation whose values are at [ns] may be given [n] values: as
   many as there are nodes, or any number if it is a handler or a way
   out that takes no more values than there are nodes. *)
Definition fits (c : cont_value) (ns : list node) (n : nat) : Prop :=
  n = length ns ∨ ∃ t : nat, target_length c = Some t ∧ t <= length ns.

(* The code of a unit with a call of unknown arity: if not tupled, it
   has parameters; if tupled, its parameters are of one component
   each. *)
Definition indirect_ok : Prop :=
  ∀ (id : code_id) (c : code),
    (id, c) ∈ codes ->
    if code_is_tupled c
    then singleton_params (code_params_arity c) = true
    else code_params_arity c <> [].

(* The values [vs], in groups by the arity [a], are at the parameters
   of the witnesses that the nodes [ws] stand for. *)
Definition args_at (ws : list node) (vs : list value) (a : arity) : Prop :=
  ∀ (i : nat) (x : value) (g k : nat) (w W p : node),
    nth_error vs i = Some x ->
    group_position (arity_sizes a) i = Some (g, k) ->
    nth_error ws g = Some w ->
    may_come_from sol W w -> g_parameter G W k p -> flows_ x p.

(* A call of unknown arity of what is at [start], with the values [vs]
   at the arity [a], by the witnesses [ws], one for each group: what it
   returns is at [returns], what it raises at [exn], and the code that
   a witness stands for is used. *)
Definition pending (start : node) (ws : list node) (vs : list value)
    (a : arity) (returns : list node) (exn : node) : Prop :=
  length ws = length a
  ∧ call_chain G sol (fun y => may_come_from sol y start) ws
  ∧ args_at ws vs a
  ∧ (∀ w : node,
       nth_error ws (pred (length ws)) = Some w ->
       (∀ (i : nat) (r W R : node) (v : value),
          nth_error returns i = Some r -> may_come_from sol W w ->
          g_constructor G W (Field_return_of_call (Normal i)) R ->
          flows_ v R -> flows_ v r)
       ∧ ∀ r y : node,
           nth_error returns 0 = Some r -> returned_at G sol w y ->
           may_come_from sol y r)
  ∧ (∀ (g : nat) (w W E : node) (v : value),
       nth_error ws g = Some w -> may_come_from sol W w ->
       g_constructor G W (Field_return_of_call Exn) E ->
       flows_ v E -> flows_ v exn)
  ∧ (∀ (g : nat) (w W : node) (id : code_id),
       nth_error ws g = Some w -> may_come_from sol W w ->
       g_constructor G W Field_code_id_of_call_witness (Node_code_id id) ->
       au (Node_code_id id)).

(* The continuations of a state, as the traversal had them: handlers,
   each traversed with the continuations of its definition, and, in
   code, the two continuations of the code, which stand for those of
   the call and are given what the code returns at its nodes; a
   continuation with extra arguments, which are at nodes after those of
   its values; and what is left of an over-application, a call of
   unknown arity of the value that it is given. *)
Inductive conts_ok : option code_id -> cont_env -> finite_map (list node)
    -> Prop :=
  | Conts_top : conts_ok None [] (unit_conts u)
  | Conts_code (id : code_id) (ret exn : continuation)
      (c_ret c_exn : cont_value) (nr : list node) (ne : node) :
    ret <> exn -> cont_ok c_ret nr -> cont_ok c_exn [ne] ->
    conts_ok (Some id) [(ret, c_ret); (exn, c_exn)] [(exn, [ne]); (ret, nr)]
  | Conts_cons (ctx : option code_id) (k : continuation)
      (h : continuation_handler) (vars : env) (K0 : cont_env)
      (recursive : list (continuation * continuation_handler))
      (K : cont_env) (ce : finite_map (list node)) :
    cont_ok (Closure h vars K0 recursive) (handler_params h) ->
    conts_ok ctx K ce ->
    conts_ok ctx ((k, Closure h vars K0 recursive) :: K)
      ((k, handler_params h) :: ce)
with cont_ok : cont_value -> list node -> Prop :=
  | Outer_ok (k : continuation) (ns : list node) :
    transfers ns (unit_cont_nodes u k) -> cont_ok (Outer k) ns
  | Closure_ok (h : continuation_handler) (vars : env) (K0 : cont_env)
      (recursive : list (continuation * continuation_handler))
      (ns : list node) (ctx : option code_id)
      (ce0 : finite_map (list node)) :
    env_flows_ vars ->
    conts_ok ctx K0 ce0 ->
    live ctx ->
    (∀ (k' : continuation) (h' : continuation_handler),
       (k', h') ∈ recursive ->
       handler_edges_ ctx (rec_conts ce0 recursive) h'
       ∧ handler_counted ctx (rec_conts ce0 recursive) h') ->
    handler_edges_ ctx (rec_conts ce0 recursive) h ->
    handler_counted ctx (rec_conts ce0 recursive) h ->
    transfers ns (handler_params h) ->
    cont_ok (Closure h vars K0 recursive) ns
  | Extra_ok (c : cont_value) (extra : list value) (ns ns' : list node) :
    cont_ok c (ns ++ ns') ->
    fits c (ns ++ ns') (length ns + length extra) ->
    (∀ (i : nat) (v : value) (n : node),
       nth_error extra i = Some v -> nth_error ns' i = Some n -> flows_ v n) ->
    cont_ok (Extra_args c extra) ns
  | Over_ok (remaining : list value) (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) (n n0 : node)
      (ns ws returns : list node) (exn : node) :
    indirect_ok ->
    (∀ v : value, flows_ v n -> flows_ v n0) ->
    pending n0 ws remaining remaining_arity returns exn ->
    cont_ok return_to returns ->
    fits return_to returns (length (unarize return_arity)) ->
    cont_ok exn_to [exn] ->
    cont_ok
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (n :: ns).

Scheme conts_ok_mut := Induction for conts_ok Sort Prop
  with cont_ok_mut := Induction for cont_ok Sort Prop.
Combined Scheme conts_ok_comb from conts_ok_mut, cont_ok_mut.

Lemma conts_ok_lookup (ctx : option code_id) (K : cont_env)
    (ce : finite_map (list node)) (k : continuation) (c : cont_value) :
  conts_ok ctx K ce -> lookup k K = Some c ->
  ∃ ns : list node,
    lookup k ce = Some ns ∧ cont_ok c ns
    ∧ (ctx = None -> target_length c = Some (length ns)).
Proof.
  intros Hok. induction Hok as [ | id ret exn c_ret c_exn nr ne Hne Hret Hexn
                               | ctx k' h vars K0 recursive K ce Hcl Hok IH];
    simpl; [ discriminate | | ].
  - destruct (Pos.eqb_spec k ret) as [-> | Hk].
    + intros [= <-]. exists nr. split; [ | split; [ exact Hret | discriminate
      ] ].
      destruct (Pos.eqb_spec ret exn); [ contradiction | reflexivity ].
    + destruct (Pos.eqb_spec k exn) as [-> | _]; [ | discriminate ].
      intros [= <-]. exists [ne].
      split; [ reflexivity | split; [ exact Hexn | discriminate ] ].
  - destruct (Pos.eqb k k'); [ | exact IH ].
    intros [= <-]. exists (handler_params h).
    split; [ reflexivity | split; [ exact Hcl | reflexivity ] ].
Qed.

Lemma conts_ok_none (ctx : option code_id) (K : cont_env)
    (ce : finite_map (list node)) (k : continuation) :
  conts_ok ctx K ce -> lookup k K = None ->
  match ctx with
  | None => lookup k ce = lookup k (unit_conts u)
  | Some _ => lookup k ce = None
  end.
Proof.
  intros Hok. induction Hok as [ | id ret exn c_ret c_exn nr ne Hne Hret Hexn
                               | ctx k' h vars K0 recursive K ce Hcl Hok IH];
    simpl; [ reflexivity | | ].
  - destruct (Pos.eqb k ret); [ discriminate | ].
    destruct (Pos.eqb k exn); [ discriminate | reflexivity ].
  - destruct (Pos.eqb k k'); [ discriminate | exact IH ].
Qed.

(* What the continuations keep is where the solution says. *)
Lemma conts_ok_flows :
  (∀ (ctx : option code_id) (K : cont_env) (ce : finite_map (list node)),
     conts_ok ctx K ce ->
     ∀ (k : continuation) (c : cont_value),
       (k, c) ∈ K -> cont_flows G sol keep called located c)
  ∧ (∀ (c : cont_value) (ns : list node),
       cont_ok c ns -> cont_flows G sol keep called located c).
Proof.
  apply conts_ok_comb.
  - intros k c [].
  - intros id ret exn c_ret c_exn nr ne _ _ IHret _ IHexn k c
      [[= <- <-] | [[= <- <-] | []]]; assumption.
  - intros ctx k h vars K0 recursive K ce _ IHcl _ IH k' c [[= <- <-] | Hin];
      [ exact IHcl | exact (IH k' c Hin) ].
  - intros k ns _. apply Outer_flows.
  - intros h vars K0 recursive ns ctx ce0 Hvars _ IH _ _ _ _ _.
    apply Closure_flows; [ exact Hvars | exact IH ].
  - intros c extra ns ns' _ IH _ _. apply Extra_args_flows. exact IH.
  - intros remaining remaining_arity return_arity return_to exn_to n n0 ns
      ws returns exn _ _ _ _ IHret _ _ IHexn.
    apply Over_application_flows; assumption.
Qed.

(* What is at nodes [ns'] that goes to [ns] goes where [ns] go. *)
Lemma transfers_compose (ns ns' targets : list node) :
  (∀ (i : nat) (n' n : node),
     nth_error ns' i = Some n' -> nth_error ns i = Some n ->
     ∀ v : value, flows_ v n' -> flows_ v n) ->
  length ns' <= length ns ∨ length targets <= length ns ->
  transfers ns targets -> transfers ns' targets.
Proof.
  intros Htr Hl Ht i v n' p Hn' Hp Hv.
  destruct (nth_error ns i) as [n | ] eqn:Hn.
  - exact (Ht i v n p Hn Hp (Htr i n' n Hn' Hn v Hv)).
  - exfalso. apply nth_error_None in Hn.
    destruct Hl as [Hl | Hl].
    + assert (i < length ns') by (apply nth_error_Some; congruence). lia.
    + assert (i < length targets) by (apply nth_error_Some; congruence).
      lia.
Qed.

Lemma cont_ok_compose_gen :
  (∀ (ctx : option code_id) (K : cont_env) (ce : finite_map (list node)),
     conts_ok ctx K ce -> True)
  ∧ (∀ (c : cont_value) (ns : list node),
       cont_ok c ns ->
       ∀ ns' : list node,
         (∀ (i : nat) (n' n : node),
            nth_error ns' i = Some n' -> nth_error ns i = Some n ->
            ∀ v : value, flows_ v n' -> flows_ v n) ->
         length ns' = length ns
         ∨ (∃ t : nat, target_length c = Some t ∧ t <= length ns) ->
         cont_ok c ns').
Proof.
  apply conts_ok_comb; try (intros; exact I).
  - intros k ns Ht ns' Htr Hlen. apply Outer_ok.
    apply (transfers_compose ns ns' _ Htr); [ | exact Ht ].
    destruct Hlen as [Hlen | (t & Ht' & Hle)]; [ left; lia | right ].
    simpl in Ht'. injection Ht' as <-. exact Hle.
  - intros h vars K0 recursive ns ctx ce0 Hvars Hconts _ Hlive Hrec Hh Hcount
      Ht ns' Htr Hlen.
    apply (Closure_ok h vars K0 recursive ns' ctx ce0); try assumption.
    apply (transfers_compose ns ns' _ Htr); [ | exact Ht ].
    destruct Hlen as [Hlen | (t & Ht' & Hle)]; [ left; lia | right ].
    simpl in Ht'. injection Ht' as <-. exact Hle.
  - intros c extra ns nsx Hc' IHc Hfits Hextra ns' Htr Hlen.
    destruct Hlen as [Hlen | (t & Ht & _)]; [ | discriminate Ht ].
    apply (Extra_ok c extra ns' nsx); [ | | exact Hextra ].
    + apply IHc.
      * intros i n' n Hn' Hn v Hv.
        destruct (Nat.ltb_spec i (length ns')) as [Hi | Hi].
        -- rewrite nth_error_app1 in Hn' by exact Hi.
           rewrite nth_error_app1 in Hn by lia.
           exact (Htr i n' n Hn' Hn v Hv).
        -- rewrite nth_error_app2 in Hn' by exact Hi.
           rewrite nth_error_app2 in Hn by lia.
           rewrite Hlen in Hn'. congruence.
      * left. rewrite !length_app, Hlen. reflexivity.
    + unfold fits in *. rewrite !length_app in *. rewrite Hlen. exact Hfits.
  - intros remaining remaining_arity return_arity return_to exn_to n n0 ns
      ws returns exn Hind Htr0 Hpend Hret _ Hfits Hexn _ ns' Htr Hlen.
    destruct Hlen as [Hlen | (t & Ht & _)]; [ | discriminate Ht ].
    destruct ns' as [ | n' ns'']; [ discriminate Hlen | ].
    apply (Over_ok remaining remaining_arity return_arity return_to exn_to n'
             n0 ns'' ws returns exn); try assumption.
    intros v Hv. apply Htr0. exact (Htr 0 n' n eq_refl eq_refl v Hv).
Qed.

(* A continuation given values at other nodes, from which they go to
   its own. *)
Lemma cont_ok_compose (c : cont_value) (ns ns' : list node) :
  cont_ok c ns ->
  (∀ (i : nat) (n' n : node),
     nth_error ns' i = Some n' -> nth_error ns i = Some n ->
     ∀ v : value, flows_ v n' -> flows_ v n) ->
  length ns' = length ns
  ∨ (∃ t : nat, target_length c = Some t ∧ t <= length ns) ->
  cont_ok c ns'.
Proof. intros H. exact (proj2 cont_ok_compose_gen c ns H ns'). Qed.

(* The closures of a recursive group in front of the continuations of
   its definition. *)
Lemma conts_ok_rec (ctx : option code_id) (vars : env) (conts : cont_env)
    (ce0 : finite_map (list node))
    (recursive : list (continuation * continuation_handler)) :
  env_flows_ vars -> conts_ok ctx conts ce0 -> live ctx ->
  (∀ (k : continuation) (h : continuation_handler),
     (k, h) ∈ recursive ->
     handler_edges_ ctx (rec_conts ce0 recursive) h
     ∧ handler_counted ctx (rec_conts ce0 recursive) h) ->
  conts_ok ctx (rec_closures recursive vars conts ++ conts)
    (rec_conts ce0 recursive).
Proof.
  intros Hvars Hconts Hlive Hrec. unfold rec_closures, rec_conts.
  assert (Hgen : ∀ l : list (continuation * continuation_handler),
            (∀ kh, kh ∈ l -> kh ∈ recursive) ->
            conts_ok ctx
              (map (fun '(k, h) => (k, Closure h vars conts recursive)) l
               ++ conts)
              (map (fun '(k, h) => (k, handler_params h)) l ++ ce0)).
  { induction l as [ | [k h] l IH]; intros Hl; [ exact Hconts | ].
    simpl. apply Conts_cons; [ | apply IH; intros kh Hkh; apply Hl; right;
                                 exact Hkh ].
    destruct (Hrec k h (Hl _ (or_introl eq_refl))) as [Hh Hcount].
    apply (Closure_ok h vars conts recursive (handler_params h) ctx ce0);
      try assumption.
    intros i v n p Hn Hp. rewrite Hn in Hp. injection Hp as <-. auto. }
  exact (Hgen recursive (fun _ H => H)).
Qed.

Lemma lookup_counts (ce : finite_map (list node)) (k : continuation) :
  lookup k (counts ce) = option_map (@length node) (lookup k ce).
Proof.
  induction ce as [ | [k' ns] ce IH]; [ reflexivity | ].
  simpl. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* In code, a continuation counted for [n] values has nodes for [n]. *)
Lemma count_ok_lookup (ce : finite_map (list node)) (k : continuation)
    (n : nat) :
  count_ok (counts ce) k n = true ->
  ∃ ns : list node, lookup k ce = Some ns ∧ length ns = n.
Proof.
  unfold count_ok. rewrite lookup_counts.
  destruct (lookup k ce) as [ns | ]; simpl; [ | discriminate ].
  intros H. apply Nat.eqb_eq in H. exists ns. auto.
Qed.

Lemma counts_rec (ce : finite_map (list node))
    (handlers : list (continuation * continuation_handler)) :
  counts (rec_conts ce handlers)
  = map (fun '(k, h) => (k, length (handler_params h))) handlers
    ++ counts ce.
Proof.
  unfold counts, rec_conts. rewrite map_app.
  apply (f_equal (fun l => l ++ _)).
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

(* The callee of a call is a closure whose allocation the call witness
   is read from. *)
Lemma closure_alloc (statics : static_env) (v : value) (n : node)
    (id : code_id) (slots : list (value_slot * value)) :
  statics_flow_ statics -> flows_ v n ->
  closure_of statics v = Some (id, slots) ->
  ∃ b : node, may_come_from sol b n ∧ flows_ v b ∧ called b id.
Proof.
  intros Hst Hv Hcl.
  destruct v as [ | code slots0 | slot decls slots0 | s | | | | | ];
    simpl in Hcl; try discriminate Hcl.
  - destruct Hv.
  - destruct (lookup slot decls) as [id0 | ] eqn:Hl; [ | discriminate Hcl ].
    injection Hcl as -> _.
    destruct Hv as (sites & Hn & Hall).
    exists (sites slot). split; [ exact Hn | split ].
    + exists sites. split; [ exact (proj1 (Hall slot id Hl)) | exact Hall ].
    + exact (proj1 (proj2 (Hall slot id Hl))).
  - destruct (lookup s (static_values statics))
      as [[ | slot syms decls slots0 | | | | | ] | ] eqn:Hls;
      try discriminate Hcl.
    + destruct (Hst s _ Hls) as [_ []].
    + destruct (lookup slot decls) as [id0 | ] eqn:Hl; [ | discriminate Hcl ].
      injection Hcl as -> _.
      destruct (Hst s _ Hls) as (Hself & Hcalled & _).
      exists (Node_symbol s). split; [ exact Hv | split ];
        [ exact Hself | exact (Hcalled id Hl) ].
Qed.

Lemma cond_alias_alias (ctx : option code_id) (to_ from : node) :
  live ctx -> cond_alias_edge G ctx to_ from -> s_alias sol to_ from = true.
Proof.
  destruct ctx as [id | ]; simpl; intros Hlive H.
  - exact (alias_propagate _ _ _ Hc _ _ _ Hlive H).
  - exact (alias_input _ _ _ Hc _ _ H).
Qed.

Lemma lookup_nodup_in {A : Type} (l : finite_map A) (k : positive) (v : A) :
  NoDup (map fst l) -> (k, v) ∈ l -> lookup k l = Some v.
Proof.
  intros Hnodup Hin. rewrite <- (app_nil_r l).
  exact (lookup_app_nodup l [] k v Hnodup Hin).
Qed.

Lemma bind_params_length (params : list variable) (vs : list value)
    (vars vars' : env) :
  bind_params params vs vars = Some vars' -> length params = length vs.
Proof.
  revert vs vars. induction params as [ | p params IH];
    intros [ | w vs] vars H; simpl in H; try discriminate; [ reflexivity | ].
  simpl. f_equal. exact (IH _ _ H).
Qed.

(* The arguments of an [Apply_cont] are at the nodes of the parameters. *)
Lemma apply_cont_args_flow (vars : env) (ce : finite_map (list node))
    (ac : apply_cont) (vs : list value) (params : list node) :
  env_flows_ vars -> apply_cont_edges G noted ce ac ->
  eval_simples vars (ac_args ac) = Some vs ->
  lookup (ac_continuation ac) ce = Some params ->
  ∀ (i : nat) (v : value) (p : node),
    nth_error vs i = Some v -> nth_error params i = Some p ->
    flows_ v p.
Proof.
  intros Hvars [Haliases Hnoted] Hvs Hparams i v p Hv Hp.
  destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
  apply (flows_alias _ _ _ _ _ _ Hc v _ (node_of_simple s)).
  - apply (alias_input _ _ _ Hc). apply (Haliases params Hparams p s).
    exact (nth_error_combine _ _ _ _ _ Hp Hs).
  - exact (simple_flows _ _ _ Hvars (Hnoted s (nth_error_In _ _ Hs)) Hsv).
Qed.

(* An arm found is a binding of the arms that counts, which the
   traversal goes through. *)
Lemma find_arm_compact (i : Z) (arms : list (Z * apply_cont))
    (ac : apply_cont) :
  find_arm i arms = Some ac -> (i, ac) ∈ Keyed.compact Z.eqb arms.
Proof.
  intros Harm. apply (keyed_in_compact Z.eqb z_eqb_decides_equality).
  rewrite <- Harm. clear Harm.
  induction arms as [ | [d ac'] arms IH]; [ reflexivity | ].
  simpl. destruct (Z.eqb i d); [ reflexivity | exact IH ].
Qed.

(* ------------------------------------------------------------------------ *)
(* The invariant                                                            *)
(* ------------------------------------------------------------------------ *)

(* The partial applications of a state are as the run makes them. *)
Definition state_pa_ok (s : state) : Prop :=
  env_pa_ok (state_statics s) (state_vars s)
  ∧ statics_pa_ok (state_statics s)
  ∧ conts_pa_ok (state_statics s) (state_conts s).

(* The invariant of the run: the facts of the expression to run are in
   the graph, in the code [ctx], which is used, with the nodes [ce] of
   the continuations in scope, as counted in code; the values of the
   state are where the solution says; and its partial applications are
   as the run makes them. *)
Definition flow_invariant (s : state) : Prop :=
  ∃ (ctx : option code_id) (ce : finite_map (list node)),
    edges ctx ce (state_expr s)
    ∧ live ctx
    ∧ counted ctx ce (state_expr s)
    ∧ env_flows_ (state_vars s)
    ∧ conts_ok ctx (state_conts s) ce
    ∧ statics_flow_ (state_statics s)
    ∧ codes_ok (state_statics s)
    ∧ state_pa_ok s.

(* What a step comes to keeps the invariant, or exits as the unit
   says. *)
Definition result_ok (r : step_result) : Prop :=
  match r with
  | Step s' => flow_invariant s'
  | Exit k vs _ _ => exit_flows G sol keep called located u k vs
  | Undefined => True
  end.

(* ------------------------------------------------------------------------ *)
(* Chains of witnesses                                                      *)
(* ------------------------------------------------------------------------ *)

Lemma call_chain_mono (P P' : node -> Prop) (ws : list node) :
  (∀ y : node, P' y -> P y) ->
  call_chain G sol P ws -> call_chain G sol P' ws.
Proof.
  destruct ws as [ | w ws]; simpl; [ auto | ].
  intros HP [Hw Hrest]. split; [ | exact Hrest ].
  intros y W Hy. exact (Hw y W (HP y Hy)).
Qed.

Lemma call_chain_app (P : node -> Prop) (l1 l2 : list node) :
  call_chain G sol P (l1 ++ l2)
  <-> call_chain G sol P l1 ∧ call_chain G sol (chain_end G sol P l1) l2.
Proof.
  revert P. induction l1 as [ | w l1 IH]; intros P; simpl.
  - tauto.
  - rewrite IH. tauto.
Qed.

Lemma chain_end_last (P : node -> Prop) (l : list node) (w y : node) :
  nth_error l (pred (length l)) = Some w ->
  chain_end G sol P l y <-> returned_at G sol w y.
Proof.
  revert P. induction l as [ | w' l IH]; intros P Hw; [ discriminate Hw | ].
  destruct l as [ | w'' l].
  - simpl in Hw. injection Hw as <-. simpl. tauto.
  - exact (IH (returned_at G sol w') Hw).
Qed.

Lemma nth_error_combine_inv {A B : Type} (l1 : list A) (l2 : list B)
    (g : nat) (a : A) (b : B) :
  nth_error (combine l1 l2) g = Some (a, b) ->
  nth_error l1 g = Some a ∧ nth_error l2 g = Some b.
Proof.
  revert l2 g. induction l1 as [ | x l1 IH]; intros [ | y l2] [ | g] H;
    simpl in *; try discriminate.
  - injection H as -> ->. auto.
  - exact (IH l2 g H).
Qed.

Lemma hd_error_combine {A B : Type} (l1 : list A) (l2 : list B) (a : A)
    (b : B) :
  hd_error (combine l1 l2) = Some (a, b) -> hd_error l2 = Some b.
Proof.
  destruct l1, l2; simpl; intros H; try discriminate. congruence.
Qed.

Lemma code_chain_nth (id : code_id) (returns : list node) (exn : node)
    (L : list (list node * node)) (g : nat) (first : list node) (W : node) :
  code_chain_edges G id returns exn L ->
  nth_error L g = Some (first, W) ->
  (∀ (k : nat) (p : node), nth_error first k = Some p -> g_parameter G W k p)
  ∧ g_constructor G W Field_code_id_of_call_witness (Node_code_id id)
  ∧ (S g = length L ->
     g_constructor G W (Field_return_of_call Exn) exn
     ∧ ∀ (i : nat) (r : node),
         nth_error returns i = Some r ->
         g_constructor G W (Field_return_of_call (Normal i)) r)
  ∧ (∀ (first' : list node) (W' : node),
       nth_error L (S g) = Some (first', W') ->
       ∃ v : node,
         g_constructor G W (Field_return_of_call (Normal 0)) v
         ∧ g_constructor G v (Field_call_witness Unknown_arity_code_pointer)
             W').
Proof.
  revert g. induction L as [ | [f w] L IH]; intros g HL Hg;
    [ destruct g; discriminate | ].
  destruct HL as (Hp & Hcode & Hnext & HL).
  destruct g as [ | g].
  - simpl in Hg. injection Hg as <- <-.
    split; [ exact Hp | split; [ exact Hcode | split ] ].
    + intros Hlen. destruct L as [ | [f' w'] L'];
        [ exact Hnext | simpl in Hlen; discriminate ].
    + intros f' W' HW'. destruct L as [ | [f'' w''] L'];
        [ discriminate HW' | ].
      simpl in HW'. injection HW' as <- <-. exact Hnext.
  - simpl in Hg. destruct (IH g HL Hg) as (H1 & H2 & H3 & H4).
    split; [ exact H1 | split; [ exact H2 | split ] ].
    + intros Hlen. apply H3. simpl in Hlen. lia.
    + intros f' W' HW'. simpl in HW'. exact (H4 f' W' HW').
Qed.

Lemma caller_chain_nth (returns : list node) (exn called_node : node)
    (L : list (list simple * node)) (g : nat) (first : list simple)
    (w : node) :
  caller_chain_edges G returns exn called_node L ->
  nth_error L g = Some (first, w) ->
  (∀ (k : nat) (s : simple),
     nth_error first k = Some s -> g_argument G (node_of_simple s) k w)
  ∧ g_accessor G exn (Field_return_of_call Exn) w
  ∧ g_accessor G called_node Field_code_id_of_call_witness w
  ∧ (S g = length L ->
     ∀ (i : nat) (r : node),
       nth_error returns i = Some r ->
       g_accessor G r (Field_return_of_call (Normal i)) w)
  ∧ (∀ (first' : list simple) (w' : node),
       nth_error L (S g) = Some (first', w') ->
       ∃ v : node,
         g_accessor G v (Field_return_of_call (Normal 0)) w
         ∧ g_accessor G w' (Field_call_witness Unknown_arity_code_pointer) v).
Proof.
  revert g. induction L as [ | [f w0] L IH]; intros g HL Hg;
    [ destruct g; discriminate | ].
  destruct HL as (Hargs & Hexn & Hcalled & Hnext & HL).
  destruct g as [ | g].
  - simpl in Hg. injection Hg as <- <-.
    split; [ exact Hargs | split; [ exact Hexn | split; [ exact Hcalled
      | split ] ] ].
    + intros Hlen. destruct L as [ | [f' w'] L'];
        [ exact Hnext | simpl in Hlen; discriminate ].
    + intros f' w' Hw'. destruct L as [ | [f'' w''] L'];
        [ discriminate Hw' | ].
      simpl in Hw'. injection Hw' as <- <-. exact Hnext.
  - simpl in Hg. destruct (IH g HL Hg) as (H1 & H2 & H3 & H4 & H5).
    split; [ exact H1 | split; [ exact H2 | split; [ exact H3 | split ] ] ].
    + intros Hlen. apply H4. simpl in Hlen. lia.
    + intros f' w' Hw'. simpl in Hw'. exact (H5 f' w' Hw').
Qed.

(* The witnesses of a call stand for those of the code that the callee
   has, group by group. *)
Lemma chain_cover (id : code_id) (returns : list node) (exn : node)
    (L : list (list node * node)) :
  code_chain_edges G id returns exn L ->
  (∀ (first : list node) (W : node), (first, W) ∈ L -> may_come_from sol W W) ->
  ∀ (P : node -> Prop) (ws : list node),
    call_chain G sol P ws ->
    (∀ (first : list node) (W : node),
       hd_error L = Some (first, W) ->
       ∃ y : node,
         P y
         ∧ g_constructor G y (Field_call_witness Unknown_arity_code_pointer)
             W) ->
    ∀ (g : nat) (w : node) (first : list node) (W : node),
      nth_error ws g = Some w -> nth_error L g = Some (first, W) ->
      may_come_from sol W w.
Proof.
  induction L as [ | [f W1] L IH];
    intros HL Hself P ws Hchain Hfirst g w first W Hw HW;
    [ destruct g; discriminate | ].
  destruct HL as (_ & _ & Hnext & HL).
  destruct ws as [ | w1 ws]; [ destruct g; discriminate | ].
  destruct Hchain as [Hcalled Hchain].
  assert (H1 : may_come_from sol W1 w1).
  { destruct (Hfirst f W1 eq_refl) as (y & Hy & Hcons).
    exact (Hcalled y W1 Hy Hcons (Hself f W1 (or_introl eq_refl))). }
  destruct g as [ | g].
  - simpl in Hw, HW. injection Hw as <-. injection HW as <- <-. exact H1.
  - simpl in Hw, HW.
    refine (IH HL (fun f' W' Hin => Hself f' W' (or_intror Hin))
              (returned_at G sol w1) ws Hchain _ g w first W Hw HW).
    intros f' W' Hhd. destruct L as [ | [f'' W''] L']; [ discriminate Hhd | ].
    simpl in Hhd. injection Hhd as -> ->.
    destruct Hnext as (v & Hv & Hvw).
    exists v. split; [ | exact Hvw ].
    exists W1, v. split; [ exact H1 | split; [ exact Hv | ] ].
    apply (alloc_self _ _ _ Hvw).
    exact (proj1 (Hself f' W' (or_intror (or_introl eq_refl)))).
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma every_app_intro {A : Type} (P : A -> Prop) (l1 l2 : list A) :
  every P l1 -> every P l2 -> every P (l1 ++ l2).
Proof.
  rewrite !(every_in A). intros H1 H2 x Hx.
  apply in_app_or in Hx as [Hx | Hx]; auto.
Qed.

Lemma every_firstn {A : Type} (P : A -> Prop) (n : nat) (l : list A) :
  every P l -> every P (firstn n l).
Proof.
  rewrite !(every_in A). intros H x Hx. apply H.
  rewrite <- (firstn_skipn n l). apply in_or_app. left. exact Hx.
Qed.

Lemma every_skipn {A : Type} (P : A -> Prop) (n : nat) (l : list A) :
  every P l -> every P (skipn n l).
Proof.
  rewrite !(every_in A). intros H x Hx. apply H.
  rewrite <- (firstn_skipn n l). apply in_or_app. right. exact Hx.
Qed.

Lemma length_unarize_app (a1 a2 : arity) :
  length (unarize (a1 ++ a2)) = length (unarize a1) + length (unarize a2).
Proof. unfold unarize. rewrite flat_map_app, length_app. reflexivity. Qed.

(* What the code bound in the static environment has, as the graph
   says. *)
Lemma code_facts (statics : static_env) (id : code_id) (c : code) :
  codes_ok statics -> lookup id (static_code statics) = Some c ->
  (id, c) ∈ codes
  ∧ ∃ cd : code_dep,
      lookup id cds = Some cd
      ∧ (code_is_tupled c = false ->
         length (cd_unknown_arity_call_witnesses cd)
         = length (at_least_one_group
                     (group_by_parameter (code_params_arity c)
                        (cd_params cd)))
         ∧ code_chain_edges G id (cd_return cd) (cd_exn cd)
             (combine
                (at_least_one_group
                   (group_by_parameter (code_params_arity c) (cd_params cd)))
                (cd_unknown_arity_call_witnesses cd)))
      ∧ (code_is_tupled c = true -> tupled_witness_edges G id cd)
      ∧ length (cd_params cd) = length (unarize (code_params_arity c))
      ∧ length (cd_return cd) = length (unarize (code_result_arity c)).
Proof.
  intros Hcodes Hcode. destruct (Hcodes id c Hcode) as [Hin Hedges].
  split; [ exact Hin | ].
  destruct c as [params mc ret exn metadata body].
  destruct Hedges as (cd & Hcd & _ & Hu & Ht & Lp & Lr & _).
  exists cd. split; [ exact (lookup_nodup_in _ _ _ Hnd Hcd) | ].
  split; [ exact Hu | split; [ exact Ht | split; [ exact Lp | exact Lr ] ] ].
Qed.

(* The state that a full call runs keeps the invariant, if the values
   and the continuations are at the nodes of the code. *)
Lemma enter_code_ok (id : code_id) (c : code) (closure : value)
    (vs : list value) (callee_vars : env) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  lookup id (static_code statics) = Some c ->
  live (Some id) ->
  length vs = length (unarize (code_params_arity c)) ->
  (∀ cd : code_dep,
     lookup id cds = Some cd ->
     (∀ (i : nat) (x : value) (p : node),
        nth_error vs i = Some x -> nth_error (cd_params cd) i = Some p ->
        flows_ x p)
     ∧ flows_ closure (cd_my_closure cd)
     ∧ cont_ok return_to (cd_return cd)
     ∧ cont_ok exn_to [cd_exn cd]) ->
  bind_params (code_params c) vs [(code_my_closure c, closure)]
  = Some callee_vars ->
  statics_flow_ statics -> codes_ok statics -> statics_pa_ok statics ->
  every (pa_ok statics) vs -> pa_ok statics closure ->
  cont_pa_ok statics return_to -> cont_pa_ok statics exn_to ->
  flow_invariant
    (State (code_body c) callee_vars
       [(code_return_continuation c, return_to);
        (code_exn_continuation c, exn_to)]
       statics hp rd).
Proof.
  intros Hcode Hlive Hlen Hcd Hbind Hst Hcodes Hspa Hvs_pa Hcl_pa Hret_pa
    Hexn_pa.
  destruct (Hcodes id c Hcode) as [_ Hedges].
  destruct c as [params mc ret exn_k metadata cbody].
  destruct Hedges as (cd & Hcd_in & _ & _ & _ & Lp & Lr & Hmc & Hps & Hne
                      & Hbinds & Hcounted_body & Hbody).
  destruct (Hcd cd (lookup_nodup_in _ _ _ Hnd Hcd_in))
    as (Hparams & Hclosure & Hret & Hexn).
  exists (Some id), (code_conts (Code params mc ret exn_k metadata cbody) cd).
  simpl. split; [ exact Hbody | split; [ exact Hlive | split ] ].
  { intros _. simpl. rewrite Lr. exact Hcounted_body. }
  split; [ | split; [ | split; [ exact Hst | split; [ exact Hcodes | ] ] ] ].
  - intros y v Hy.
    destruct (bind_params_lookup _ _ _ _ _ _ Hbind Hy)
      as [Hold | (i & Hy_nth & Hv_nth)].
    + simpl in Hold. destruct (Pos.eqb_spec y mc) as [-> | _];
        [ | discriminate Hold ].
      injection Hold as <-.
      exact (flows_alias _ _ _ _ _ _ Hc _ _ _ (alias_input _ _ _ Hc _ _ Hmc)
               Hclosure).
    + destruct (nth_error (cd_params cd) i) as [p | ] eqn:Hp.
      * apply (flows_alias _ _ _ _ _ _ Hc v _ p).
        -- apply (alias_input _ _ _ Hc). apply Hps.
           exact (nth_error_combine _ _ _ _ _ Hy_nth Hp).
        -- exact (Hparams i v p Hv_nth Hp).
      * exfalso. apply nth_error_None in Hp.
        assert (i < length vs) by (apply nth_error_Some; congruence).
        simpl in Hlen. lia.
  - apply Conts_code; [ exact Hne | exact Hret | exact Hexn ].
  - split; [ | split; [ exact Hspa | ] ].
    + intros y v Hy.
      destruct (bind_params_lookup _ _ _ _ _ _ Hbind Hy)
        as [Hold | (i & _ & Hv_nth)].
      * simpl in Hold. destruct (Pos.eqb y mc); [ | discriminate Hold ].
        injection Hold as <-. exact Hcl_pa.
      * exact (proj1 (every_in _ _ _) Hvs_pa v (nth_error_In _ _ Hv_nth)).
    + intros k c' [[= <- <-] | [[= <- <-] | []]]; assumption.
Qed.

Lemma call_code_not_exit (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (k : continuation)
    (vs' : list value)
    (hp' : heap) (rd' : read_record) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  <> Exit k vs' hp' rd'.
Proof.
  unfold call_code. destruct (binds_statics (code_body c)); [ discriminate | ].
  destruct (negb _); [ discriminate | ].
  destruct (bind_params _ _ _); discriminate.
Qed.

(* A direct call of the fragment goes wrong, or runs the body of the code
   of its callee, which the unit binds. *)
Lemma call_step (ap : apply_expr) (vars : env) (K : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (id : code_id) :
  ap_call_kind ap = Direct id ->
  apply_in_flow_fragment codes ap = true -> codes_ok statics ->
  match step (State (Apply ap) vars K statics hp rd) with
  | Step s' =>
    ∃ (callee : value) (vs extra : list value)
      (slots : list (value_slot * value)) (c : code) (callee_vars : env),
      eval_simple vars (ap_callee ap) = Some callee
      ∧ eval_simples vars (ap_args ap) = Some vs
      ∧ eval_simples vars (exn_extra_simples ap) = Some extra
      ∧ closure_of statics callee = Some (id, slots)
      ∧ lookup id (static_code statics) = Some c
      ∧ kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap)) = true
      ∧ kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
          (unarize_per_parameter (code_params_arity c)) = true
      ∧ arity_eqb (ap_return_arity ap) (code_result_arity c) = true
      ∧ bind_params (code_params c) vs [(code_my_closure c, callee)]
        = Some callee_vars
      ∧ s' = State (code_body c) callee_vars
               [(code_return_continuation c, resolve (ap_continuation ap) K);
                (code_exn_continuation c,
                 with_extra_args (resolve (ap_exn_continuation ap) K) extra)]
               statics hp rd
  | Exit _ _ _ _ => False
  | Undefined => True
  end.
Proof.
  intros Hkind Hfrag Hcodes. pose proof Hfrag as Hall.
  unfold apply_in_flow_fragment in Hall. rewrite Hkind in Hall.
  apply andb_true_iff in Hall as [Hall _].
  rewrite step_apply.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
    [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs; [ | exact I ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
    eqn:Hextra; [ | exact I ].
  unfold call_function. rewrite Hkind, shape_of_call_direct.
  destruct (closure_of statics callee) as [[called' slots] | ] eqn:Hclosure;
    [ | unfold shape_of_closure_call; rewrite Hclosure; exact I ].
  destruct (Pos.eqb_spec called' id) as [-> | Hne];
    [ | unfold shape_of_closure_call; rewrite Hclosure;
        apply Pos.eqb_neq in Hne; rewrite Hne; exact I ].
  destruct (lookup id (static_code statics)) as [c | ] eqn:Hcode;
    [ | unfold shape_of_closure_call; rewrite Hclosure, Pos.eqb_refl, Hcode;
        exact I ].
  destruct (Hcodes id c Hcode) as [Hin _].
  rewrite forallb_forall in Hall. specialize (Hall (id, c) Hin).
  simpl in Hall. rewrite Pos.eqb_refl in Hall. simpl in Hall.
  apply andb_true_iff in Hall as [Htupled Hfull].
  apply negb_true_iff in Htupled. apply Nat.eqb_eq in Hfull.
  rewrite (shape_of_closure_call_full statics callee (Direct id) vs
             (ap_args_arity ap) (ap_return_arity ap) id slots c Hclosure
             (or_intror eq_refl) Hcode Htupled Hfull).
  simpl. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct (kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap)))
    eqn:Hk1; [ | exact I ].
  destruct (kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
              (unarize_per_parameter (code_params_arity c))) eqn:Hk2;
    [ | exact I ].
  destruct (arity_eqb (ap_return_arity ap) (code_result_arity c)) eqn:Hk3;
    [ | exact I ].
  simpl.
  destruct (bind_params (code_params c) vs [(code_my_closure c, callee)])
    as [callee_vars | ] eqn:Hbind; [ | exact I ].
  exists callee, vs, extra, slots, c, callee_vars. auto 12.
Qed.

(* The state that a direct call runs keeps the invariant. *)
Lemma direct_call_ok (ctx : option code_id) (ap : apply_expr) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (id : code_id)
    (callee : value)
    (vs : list value) (slots : list (value_slot * value)) (c : code)
    (callee_vars : env) (returns : list node) (exn : node)
    (return_to exn_to : cont_value) :
  ap_call_kind ap = Direct id ->
  call_edges G ctx ap returns exn ->
  simple_noted noted (ap_callee ap) -> simples_noted noted (ap_args ap) ->
  live ctx -> env_flows_ vars -> env_pa_ok statics vars ->
  statics_flow_ statics -> codes_ok statics -> statics_pa_ok statics ->
  cont_ok return_to returns ->
  fits return_to returns (length (unarize (ap_return_arity ap))) ->
  cont_pa_ok statics return_to ->
  cont_ok exn_to [exn] -> cont_pa_ok statics exn_to ->
  eval_simple vars (ap_callee ap) = Some callee ->
  eval_simples vars (ap_args ap) = Some vs ->
  closure_of statics callee = Some (id, slots) ->
  lookup id (static_code statics) = Some c ->
  kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap)) = true ->
  kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
    (unarize_per_parameter (code_params_arity c)) = true ->
  arity_eqb (ap_return_arity ap) (code_result_arity c) = true ->
  bind_params (code_params c) vs [(code_my_closure c, callee)]
  = Some callee_vars ->
  flow_invariant
    (State (code_body c) callee_vars
       [(code_return_continuation c, return_to);
        (code_exn_continuation c, exn_to)]
       statics hp rd).
Proof.
  intros Hk Hedges Hnoted_callee Hnoted_args Hlive Hvars Hvars_pa Hst Hcodes
    Hspa Hret Hfits Hret_pa Hexn Hexn_pa Hcallee Hvs Hclosure Hcode Hkinds
    Hparams Hresult Hbind.
  unfold call_edges in Hedges. rewrite Hk in Hedges.
  destruct Hedges as (witness & apply_node & called_node & Hargs & Hretacc
                      & Hexnacc & Hcalled & Hused & Hcond & Happly).
  destruct (code_facts _ _ _ Hcodes Hcode)
    as (_ & cd & Hcd & _ & _ & Lp & Lr).
  destruct (Hcds id cd Hcd) as (Hpar & Hretc & Hexnc & Hcodeid & Hasid).
  set (w := cd_known_arity_call_witness cd) in *.
  pose proof (simple_flows _ _ _ Hvars Hnoted_callee Hcallee) as Hfcallee.
  destruct (closure_alloc _ _ _ _ _ Hst Hfcallee Hclosure)
    as (b & Hb & Hfb & Hcalled_b).
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (Hprop & Hcw & _).
  assert (Hw_self : may_come_from sol w w).
  { apply (alloc_self _ _ _ Hcodeid).
    exact (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hasid)). }
  assert (Hw : may_come_from sol w witness).
  { apply (may_come_from_alias _ _ _ Hc _ _ apply_node);
      [ exact (cond_alias_alias _ _ _ Hlive Hcond) | ].
    refine (may_come_from_read _ _ _ Hc w b _ _ _ _ _ Hcw Happly Hb Hw_self).
    not_local_field. }
  assert (Hlive' : live (Some id)).
  { simpl.
    refine (read_any_usage _ _ _ Hc w witness called_node _ _ _ Hcodeid Hcalled
              Hw (any_usage_input _ _ _ Hc _ Hused)).
    not_local_field. }
  apply kinds_eqb_eq in Hkinds. apply kinds_per_parameter_eqb_eq in Hparams.
  apply unarize_per_parameter_eq in Hparams. apply arity_eqb_eq in Hresult.
  apply (enter_code_ok id c callee vs callee_vars return_to exn_to statics hp rd
           Hcode Hlive'); try assumption.
  - rewrite <- (length_map kind_of_value vs), Hkinds, Hparams. reflexivity.
  - intros cd' Hcd'. rewrite Hcd in Hcd'. injection Hcd' as <-.
    split; [ | split; [ | split ] ].
    + intros i x p Hx Hp.
      destruct (eval_simples_nth _ _ _ _ _ Hvs Hx) as (s & Hs & Hsx).
      pose proof (simple_flows _ _ _ Hvars (Hnoted_args s (nth_error_In _ _ Hs))
                    Hsx) as Hfs.
      refine (argument_transfer _ _ _ Hc (flows_ x) w witness p
                (node_of_simple s) i _ _ Hw (Hargs i s Hs) (Hpar i p Hp)
                (flows_hs _ _ _ _ _ _ Hc _ _ Hfs) Hfs).
      * intros x' y'. exact (flows_alias _ _ _ _ _ _ Hc x x' y').
      * intros x' m. exact (flows_escape _ _ _ _ _ _ Hc x x' m).
    + apply (flows_alias _ _ _ _ _ _ Hc callee _ b
               (alias_propagate _ _ _ Hc _ _ _ Hlive' Hprop)).
      exact Hfb.
    + apply (cont_ok_compose return_to returns (cd_return cd) Hret).
      * intros i r' r Hr' Hr v Hv.
        refine (flows_read _ _ _ _ _ _ Hc w witness r r' _ v _ (Hretc i r' Hr')
                  (Hretacc i r Hr) Hw Hv).
        not_local_field.
      * destruct Hfits as [Hf | Hf]; [ left | right; exact Hf ].
        rewrite Lr, <- Hresult, Hf. reflexivity.
    + apply (cont_ok_compose exn_to [exn] [cd_exn cd] Hexn);
        [ | left; reflexivity ].
      intros [ | i] n' n Hn' Hn v Hv; [ | destruct i; discriminate Hn' ].
      injection Hn' as <-. injection Hn as <-.
      refine (flows_read _ _ _ _ _ _ Hc w witness exn (cd_exn cd) _ v _ Hexnc
                Hexnacc Hw Hv).
      not_local_field.
  - exact (simples_pa_ok _ _ _ _ Hvars_pa Hvs).
  - exact (simple_pa_ok _ _ _ _ Hvars_pa Hcallee).
Qed.

(* The values of a call, together with those that a partial application
   keeps. *)
Lemma args_at_app (ws1 ws2 : list node) (vs1 vs2 : list value)
    (a1 a2 : arity) :
  length ws1 = length a1 -> length vs1 = length (unarize a1) ->
  args_at ws1 vs1 a1 -> args_at ws2 vs2 a2 ->
  args_at (ws1 ++ ws2) (vs1 ++ vs2) (a1 ++ a2).
Proof.
  intros Lw Lv H1 H2 i x g kk w W p Hx Hpos Hw HW Hp.
  rewrite arity_sizes_app in Hpos.
  destruct (Nat.ltb_spec i (list_sum (arity_sizes a1))) as [Hi | Hi].
  - rewrite group_position_app_l in Hpos by exact Hi.
    destruct (group_position_lt _ _ _ _ Hpos) as [Hg _].
    rewrite length_arity_sizes in Hg. rewrite sum_arity_sizes in Hi.
    rewrite nth_error_app1 in Hx by lia.
    rewrite nth_error_app1 in Hw by lia.
    exact (H1 i x g kk w W p Hx Hpos Hw HW Hp).
  - rewrite group_position_app_r in Hpos by exact Hi.
    destruct (group_position (arity_sizes a2) (i - list_sum (arity_sizes a1)))
      as [[g' kk'] | ] eqn:E; [ | discriminate ].
    injection Hpos as <- <-.
    rewrite length_arity_sizes in *. rewrite sum_arity_sizes in E, Hi.
    rewrite nth_error_app2 in Hx by lia. rewrite nth_error_app2 in Hw by lia.
    rewrite Lv in Hx. replace (length a1 + g' - length ws1) with g' in Hw
      by lia.
    exact (H2 _ x g' kk' w W p Hx E Hw HW Hp).
Qed.

(* What an over-application has yet to give. *)
Lemma args_at_skipn (ws : list node) (vs : list value) (a : arity)
    (k : nat) :
  args_at ws vs a ->
  args_at (skipn k ws) (skipn (list_sum (firstn k (arity_sizes a))) vs)
    (skipn k a).
Proof.
  intros H i x g kk w W p Hx Hpos Hw HW Hp.
  rewrite nth_error_skipn_add in Hx. rewrite nth_error_skipn_add in Hw.
  rewrite arity_sizes_skipn in Hpos.
  set (S := arity_sizes a) in *.
  destruct (le_lt_dec k (length S)) as [Hk | Hk];
    [ | rewrite skipn_all2 in Hpos by lia; discriminate Hpos ].
  apply (H _ x (k + g) kk w W p Hx); [ | exact Hw | exact HW | exact Hp ].
  transitivity (group_position (firstn k S ++ skipn k S)
                  (list_sum (firstn k S) + i));
    [ rewrite firstn_skipn; reflexivity | ].
  rewrite group_position_app_r by lia.
  replace (list_sum (firstn k S) + i - list_sum (firstn k S)) with i by lia.
  rewrite Hpos. rewrite length_firstn. f_equal. f_equal. lia.
Qed.

Lemma nth_error_skipn_last {A : Type} (l : list A) (d : nat) :
  d < length l ->
  nth_error (skipn d l) (pred (length (skipn d l)))
  = nth_error l (pred (length l)).
Proof.
  intros Hd. rewrite nth_error_skipn_add, length_skipn. f_equal. lia.
Qed.

(* The number of parameters of code. *)
Abbreviation nparams c := (length (code_params_arity c)).

(* A call of a closure, with the arguments that a partial application of
   it keeps ([applied], at [m], by the witnesses [ws_p]) and those of the
   call ([vs], at [start], by [ws]): a full call, an over-application,
   whose result is called with the arguments left, or a partial
   application, which goes to the return continuation. *)
Lemma closure_call_ok (closure : value) (m start : node)
    (applied vs : list value) (applied_arity args_arity return_arity : arity)
    (ws_p ws returns : list node) (exn : node)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record)
    (shape : call_shape) :
  indirect_ok ->
  flows_ closure m -> pa_ok statics closure ->
  length ws_p = length applied_arity ->
  call_chain G sol (fun y => may_come_from sol y m) ws_p ->
  args_at ws_p applied applied_arity ->
  (∀ y : node,
     chain_end G sol (fun y => may_come_from sol y m) ws_p y ->
     may_come_from sol y start) ->
  (∀ (id : code_id) (slots : list (value_slot * value)) (c : code),
     closure_of statics closure = Some (id, slots) ->
     lookup id (static_code statics) = Some c ->
     length applied_arity < nparams c ∧ code_is_tupled c = false) ->
  args_arity <> [] ->
  pending start ws vs args_arity returns exn ->
  length applied = length (unarize applied_arity)
  ∨ length vs = length (unarize args_arity) ->
  every (pa_ok statics) applied -> every (pa_ok statics) vs ->
  cont_ok return_to returns ->
  fits return_to returns (length (unarize return_arity)) ->
  cont_pa_ok statics return_to ->
  cont_ok exn_to [exn] -> cont_pa_ok statics exn_to ->
  statics_flow_ statics -> codes_ok statics -> statics_pa_ok statics ->
  (∀ v : value,
     (∀ r : node, nth_error returns 0 = Some r -> flows_ v r) ->
     pa_ok statics v -> length (unarize return_arity) = 1 ->
     result_ok (apply_cont_value return_to [v] statics hp rd)) ->
  shape_of_closure_call statics closure Indirect_unknown_arity (applied ++ vs)
    (applied_arity ++ args_arity) return_arity = Some shape ->
  result_ok
    (run_call (fun partial => apply_cont_value return_to [partial] statics hp
      rd)
       (Some shape) return_arity return_to exn_to statics hp rd).
Proof.
  intros Hind Hm Hcl_pa Hlen_p Hchain_p Hargs_p Hend_p Hjk Hne Hpend Hlen_vs
    Happ_pa Hvs_pa Hret Hfits Hret_pa Hexn Hexn_pa Hst Hcodes Hspa Hpartial
    Hshape.
  destruct Hpend as (Hlen_ws & Hchain & Hargs & Hreturns & Hraises & Huses).
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (id & slots & c & vs' & args_arity' & Hclosure & Hcode & _ & Hafc
        & Hcall).
  destruct (code_facts _ _ _ Hcodes Hcode)
    as (Hin_codes & cd & Hcd & Hunk & _ & Lp & Lr).
  destruct (Hjk id slots c Hclosure Hcode) as [Hjk' Htupled].
  pose proof (Hind id c Hin_codes) as Hparams_ne.
  rewrite Htupled in Hparams_ne.
  rewrite (arguments_for_code_not_tupled _ _ _ _ Htupled) in Hafc.
  injection Hafc as <- <-.
  destruct (Hunk Htupled) as [Lu Hcch].
  rewrite (group_by_parameter_groups _ _ Hparams_ne) in Lu, Hcch.
  rewrite length_group_by_sizes, length_arity_sizes in Lu.
  set (groups := group_by_sizes (arity_sizes (code_params_arity c))
                   (cd_params cd)) in *.
  set (L := combine groups (cd_unknown_arity_call_witnesses cd)) in *.
  set (A := applied_arity ++ args_arity) in *.
  set (all := applied ++ vs) in *.
  assert (HlenA : length A = length applied_arity + length args_arity)
    by (unfold A; apply length_app).
  assert (LL : length L = (nparams c)).
  { unfold L. rewrite length_combine, Lu. unfold groups.
    rewrite length_group_by_sizes, length_arity_sizes. lia. }
  (* The witnesses of the code are allocations of their own. *)
  destruct (Hcds id cd Hcd) as (_ & _ & _ & _ & Has_id).
  assert (Hself : ∀ (f : list node) (W : node),
            (f, W) ∈ L -> may_come_from sol W W).
  { intros f W Hin. apply In_nth_error in Hin as [g Hg].
    destruct (code_chain_nth _ _ _ _ _ _ _ Hcch Hg) as (_ & Hcodeid & _).
    apply (alloc_self _ _ _ Hcodeid).
    exact (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Has_id)). }
  (* The allocation of the closure. *)
  destruct (closure_alloc _ _ _ _ _ Hst Hm Hclosure)
    as (b & Hb & Hfb & Hcalled_b).
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (Hprop & _ & Hcw).
  (* The witnesses of the call stand for those of the code. *)
  set (wsA := ws_p ++ ws).
  assert (HchainA : call_chain G sol (fun y => may_come_from sol y m) wsA).
  { apply call_chain_app. split; [ exact Hchain_p | ].
    exact (call_chain_mono _ _ _ Hend_p Hchain). }
  assert (Hcover : ∀ (g : nat) (w : node) (f : list node) (W : node),
            nth_error wsA g = Some w -> nth_error L g = Some (f, W) ->
            may_come_from sol W w).
  { apply (chain_cover id (cd_return cd) (cd_exn cd) L Hcch Hself _ wsA
             HchainA).
    intros f W Hhd. exists b. split; [ exact Hb | ]. apply Hcw.
    exact (hd_error_combine _ _ _ _ Hhd). }
  assert (Hnth_L : ∀ g : nat,
            g < nparams c -> ∃ f W, nth_error L g = Some (f, W)).
  { intros g Hg. destruct (nth_error L g) as [[f W] | ] eqn:E; [ eauto | ].
    apply nth_error_None in E. lia. }
  assert (Hlen_wsA : length wsA = length A).
  { unfold wsA, A. rewrite !length_app, Hlen_p, Hlen_ws. reflexivity. }
  assert (HwsA : ∀ g : nat,
            length applied_arity <= g ->
            nth_error wsA g = nth_error ws (g - length applied_arity)).
  { intros g Hg. unfold wsA. rewrite nth_error_app2 by lia.
    rewrite Hlen_p. reflexivity. }
  (* The code is used. *)
  assert (Hlive : live (Some id)).
  { destruct (Hnth_L (length applied_arity) Hjk') as (f & W & HLj).
    destruct ws as [ | w0 ws'];
      [ destruct args_arity; [ contradiction | discriminate Hlen_ws ] | ].
    assert (Hw0 : nth_error wsA (length applied_arity) = Some w0).
    { rewrite HwsA by lia. rewrite Nat.sub_diag. reflexivity. }
    destruct (code_chain_nth _ _ _ _ _ _ _ Hcch HLj) as (_ & Hcodeid & _).
    exact (Huses 0 w0 W id eq_refl
             (Hcover (length applied_arity) w0 f W Hw0 HLj) Hcodeid). }
  (* The last witness of the call. *)
  assert (Hwl0 : ∃ wl : node, nth_error ws (pred (length ws)) = Some wl).
  { destruct (nth_error ws (pred (length ws))) as [wl | ] eqn:E;
      [ eauto | exfalso ].
    apply nth_error_None in E.
    destruct ws; [ destruct args_arity; [ contradiction | discriminate Hlen_ws ]
                 | simpl in E; lia ]. }
  destruct Hwl0 as (wl & Hwl).
  (* The arguments, together. *)
  assert (HargsA : length all = length (unarize A) -> args_at wsA all A).
  { intros Hl. apply args_at_app; [ exact Hlen_p | | exact Hargs_p
                                  | exact Hargs ].
    unfold all, A in Hl. rewrite length_app, length_unarize_app in Hl.
    destruct Hlen_vs; lia. }
  assert (Hparams_flow : args_at wsA all A ->
            (∀ i g kk : nat,
               group_position (arity_sizes (code_params_arity c)) i
               = Some (g, kk) ->
               group_position (arity_sizes A) i = Some (g, kk)) ->
            ∀ (i : nat) (x : value) (p : node),
              nth_error all i = Some x -> nth_error (cd_params cd) i = Some p ->
              flows_ x p).
  { intros HA Hpos i x p Hx Hp.
    assert (Hi : i < list_sum (arity_sizes (code_params_arity c))).
    { rewrite sum_arity_sizes, <- Lp. apply nth_error_Some. congruence. }
    destruct (group_position_some _ _ Hi) as (g & kk & Hg).
    destruct (group_by_sizes_nth _ _ _ _ _ _ Hg Hp) as (grp & Hgrp & Hp').
    destruct (group_position_lt _ _ _ _ Hg) as [Hgk _].
    rewrite length_arity_sizes in Hgk.
    destruct (Hnth_L g Hgk) as (f & W & HLg).
    destruct (nth_error_combine_inv _ _ _ _ _ HLg) as [Hf _].
    fold groups in Hgrp. rewrite Hgrp in Hf. injection Hf as <-.
    destruct (code_chain_nth _ _ _ _ _ _ _ Hcch HLg) as (Hpar & _).
    pose proof (Hpos i g kk Hg) as HgA.
    destruct (group_position_lt _ _ _ _ HgA) as [HgA' _].
    rewrite length_arity_sizes in HgA'.
    destruct (nth_error wsA g) as [w | ] eqn:Hw;
      [ | apply nth_error_None in Hw; lia ].
    exact (HA i x g kk w W p Hx HgA Hw (Hcover g w grp W Hw HLg)
             (Hpar kk p Hp')). }
  (* The last witness of the code. *)
  destruct (Hnth_L (pred (nparams c)) ltac:(lia)) as (fk & Wk & HLk).
  destruct (code_chain_nth _ _ _ _ _ _ _ Hcch HLk)
    as (_ & _ & Hlastf & _).
  destruct (Hlastf ltac:(lia)) as [Hexnc Hretc].
  assert (Hexn_code : pred (nparams c) < length wsA ->
            cont_ok exn_to [cd_exn cd]).
  { intros Hk. apply (cont_ok_compose exn_to [exn] [cd_exn cd] Hexn);
      [ | left; reflexivity ].
    intros [ | i] n' n Hn' Hn v Hv; [ | destruct i; discriminate Hn' ].
    injection Hn' as <-. injection Hn as <-.
    destruct (nth_error wsA (pred (nparams c))) as [wk | ] eqn:Hwk;
      [ | apply nth_error_None in Hwk; lia ].
    pose proof (Hcover _ wk fk Wk Hwk HLk) as HWk.
    rewrite HwsA in Hwk by lia.
    exact (Hraises _ wk Wk _ v Hwk HWk Hexnc Hv). }
  assert (Hm_closure : flows_ closure (cd_my_closure cd)).
  { apply (flows_alias _ _ _ _ _ _ Hc closure _ b
             (alias_propagate _ _ _ Hc _ _ _ Hlive Hprop)).
    exact Hfb. }
  assert (Hall_pa : every (pa_ok statics) all)
    by exact (every_app_intro _ _ _ Happ_pa Hvs_pa).
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[Hfull ->] | [(Hover & Hkinds & Hpoo & ->)
                     | (Hpart & Hkinds & Hpoo & _ & ->)]].
  - (* A full call. *)
    unfold A in Hfull. rewrite length_app in Hfull.
    cbn [run_call].
    destruct (call_code closure c all A return_arity return_to exn_to statics
                hp rd) as [s' | k' vs'' hp'' rd'' | ] eqn:Hcc;
      [ | exfalso; exact (call_code_not_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcc)
      | exact I ].
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ _ Hcc)
      as (callee_vars & _ & Hk1 & Hk2 & Hk3 & Hbind & ->).
    assert (HA : args_at wsA all A).
    { apply HargsA. rewrite <- (length_map kind_of_value all), Hk1.
      reflexivity. }
    apply (enter_code_ok id c closure all callee_vars return_to exn_to statics
             hp rd Hcode Hlive); try assumption.
    + rewrite <- (length_map kind_of_value all), Hk1.
      rewrite (unarize_per_parameter_eq _ _ Hk2). reflexivity.
    + intros cd' Hcd'. rewrite Hcd in Hcd'. injection Hcd' as <-.
      split; [ | split; [ exact Hm_closure | split ] ].
      * apply (Hparams_flow HA). intros i g kk H. unfold arity_sizes.
        rewrite Hk2. exact H.
      * apply (cont_ok_compose return_to returns (cd_return cd) Hret).
        -- intros i r' r Hr' Hr v Hv.
           apply (proj1 (Hreturns wl Hwl) i r Wk r' v Hr); [ | | exact Hv ].
           ++ apply (Hcover (pred (nparams c)) wl fk Wk); [ | exact HLk ].
              rewrite HwsA by lia. rewrite <- Hwl. f_equal. lia.
           ++ exact (Hretc i r' Hr').
        -- destruct Hfits as [Hf | Hf]; [ left | right; exact Hf ].
           rewrite Lr, <- Hk3, Hf. reflexivity.
      * apply Hexn_code. lia.
  - (* An over-application. *)
    cbn [run_call].
    set (size := cardinal_unarized (code_params_arity c)) in *.
    destruct (call_code closure c (firstn size all) (firstn (nparams c) A)
                (code_result_arity c)
                (Over_application (skipn size all)
                   (partially_apply A (nparams c))
                   return_arity return_to exn_to)
                exn_to statics hp rd) as [s' | k' vs'' hp'' rd'' | ] eqn:Hcc;
      [ | exfalso; exact (call_code_not_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcc)
      | exact I ].
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ _ Hcc)
      as (callee_vars & _ & Hk1 & Hk2 & _ & Hbind & ->).
    assert (HA : args_at wsA all A).
    { apply HargsA. rewrite <- (length_map kind_of_value all), Hkinds.
      reflexivity. }
    assert (Hsizes : arity_sizes (code_params_arity c)
                     = firstn (nparams c) (arity_sizes A)).
    { rewrite <- arity_sizes_firstn. unfold arity_sizes. rewrite Hk2.
      reflexivity. }
    assert (Hsize : size = list_sum (firstn (nparams c) (arity_sizes A))).
    { unfold size, cardinal_unarized. rewrite <- sum_arity_sizes, Hsizes.
      reflexivity. }
    assert (Hlen_all : length all = list_sum (arity_sizes A)).
    { rewrite sum_arity_sizes, <- (length_map kind_of_value all), Hkinds.
      reflexivity. }
    assert (Hsize_le : size <= length all).
    { rewrite Hsize, Hlen_all.
      rewrite <- (firstn_skipn (nparams c) (arity_sizes A)) at 2.
      rewrite list_sum_app. lia. }
    (* The code returns one value. *)
    unfold partial_or_over_application in Hpoo.
    apply andb_true_iff in Hpoo as [_ Hpoo].
    destruct (Nat.ltb_spec (nparams c) (length A)) as [_ | Hge]; [ | lia ].
    apply is_one_param_of_kind_value_eq in Hpoo.
    rewrite Hpoo in Lr. simpl in Lr.
    destruct (cd_return cd) as [ | R0 [ | R1 rr]] eqn:Ecr;
      try discriminate Lr.
    assert (Hk_le : pred (nparams c) < length wsA) by (rewrite Hlen_wsA; lia).
    destruct (nth_error wsA (pred (nparams c))) as [wk | ] eqn:Hwk;
      [ | apply nth_error_None in Hwk; lia ].
    assert (HWk : may_come_from sol Wk wk) by exact (Hcover _ wk fk Wk Hwk HLk).
    set (d := (nparams c) - (length applied_arity)).
    assert (Hskip : skipn (nparams c) wsA = skipn d ws).
    { unfold wsA. rewrite skipn_app, skipn_all2 by lia. simpl.
      rewrite Hlen_p. reflexivity. }
    apply (enter_code_ok id c closure (firstn size all) callee_vars
             (Over_application (skipn size all) (partially_apply A (nparams c))
                return_arity return_to exn_to)
             exn_to statics hp rd Hcode Hlive); try assumption.
    + rewrite length_firstn, Nat.min_l by exact Hsize_le. reflexivity.
    + intros cd' Hcd'. rewrite Hcd in Hcd'. injection Hcd' as <-.
      split; [ | split; [ exact Hm_closure | split ] ].
      * intros i x p Hx Hp.
        assert (Hi : i < size).
        { assert (i < length (firstn size all))
            by (apply nth_error_Some; congruence).
          rewrite length_firstn in H. lia. }
        rewrite nth_error_firstn_lt in Hx by exact Hi.
        refine (Hparams_flow HA _ i x p Hx Hp).
        intros i' g kk H. rewrite Hsizes in H.
        rewrite <- (firstn_skipn (nparams c) (arity_sizes A)).
        rewrite group_position_app_l; [ exact H | ].
        exact (proj2 (group_position_lt _ _ _ _ H)).
      * rewrite Ecr.
        apply (Over_ok (skipn size all) (partially_apply A (nparams c))
                 return_arity
                 return_to exn_to R0 R0 [] (skipn d ws) returns exn Hind);
          [ intros v Hv; exact Hv | | exact Hret | exact Hfits | exact Hexn ].
        split; [ | split; [ | split; [ | split; [ | split ] ] ] ].
        -- rewrite length_skipn. unfold partially_apply.
           rewrite length_skipn. unfold A. rewrite length_app.
           rewrite Hlen_ws. lia.
        -- rewrite <- Hskip.
           rewrite <- (firstn_skipn (nparams c) wsA) in HchainA.
           apply call_chain_app in HchainA as [_ HchainR].
           refine (call_chain_mono _ _ _ _ HchainR).
           intros y Hy.
           refine (proj2 (chain_end_last _ _ wk y _) _).
           ++ rewrite length_firstn, nth_error_firstn_lt by lia.
              rewrite <- Hwk. f_equal. lia.
           ++ exists Wk, R0. split; [ exact HWk | split; [ | exact Hy ] ].
              exact (Hretc 0 R0 eq_refl).
        -- rewrite <- Hskip. unfold partially_apply.
           rewrite Hsize. exact (args_at_skipn _ _ _ (nparams c) HA).
        -- intros wl' Hwl'. rewrite nth_error_skipn_last in Hwl' by lia.
           exact (Hreturns wl' Hwl').
        -- intros g w W E v Hw HW HE Hv.
           rewrite nth_error_skipn_add in Hw.
           exact (Hraises _ _ _ _ _ Hw HW HE Hv).
        -- intros g w W id' Hw HW Hcodeid.
           rewrite nth_error_skipn_add in Hw.
           exact (Huses _ _ _ _ Hw HW Hcodeid).
      * apply Hexn_code. exact Hk_le.
    + exact (every_firstn _ _ _ Hall_pa).
    + apply Over_pa; [ exact (every_skipn _ _ _ Hall_pa) | exact Hret_pa
                     | exact Hexn_pa ].
  - (* A partial application. *)
    cbn [run_call].
    assert (Hargs_pos : 0 < length args_arity)
      by (destruct args_arity; [ contradiction | simpl; lia ]).
    assert (HA : args_at wsA all A).
    { apply HargsA. rewrite <- (length_map kind_of_value all), Hkinds.
      reflexivity. }
    assert (Hlast : nth_error wsA (pred (length wsA)) = Some wl).
    { rewrite HwsA by (rewrite Hlen_wsA; lia). rewrite <- Hwl. f_equal.
      rewrite Hlen_wsA, Hlen_ws. lia. }
    apply Hpartial.
    + intros r Hr. simpl. exists m, wsA.
      split; [ exact Hm | split; [ exact Hlen_wsA | split; [ exact HchainA
        | split; [ | split ] ] ] ].
      * set (g := pred (length A)).
        destruct (Hnth_L g ltac:(unfold g; lia)) as (fg & Wg & HLg).
        destruct (Hnth_L (S g) ltac:(unfold g; lia)) as (fg' & Wg' & HLg').
        destruct (code_chain_nth _ _ _ _ _ _ _ Hcch HLg)
          as (_ & _ & _ & Hnext).
        destruct (Hnext fg' Wg' HLg') as (v & Hv & Hvw).
        exists v. refine (proj2 (chain_end_last _ _ wl v Hlast) _).
        exists Wg, v. split; [ | split; [ exact Hv | ] ].
        -- apply (Hcover g wl fg Wg); [ | exact HLg ].
           rewrite <- Hlast. f_equal. unfold g. rewrite Hlen_wsA. reflexivity.
        -- apply (alloc_self _ _ _ Hvw).
           exact (proj1 (Hself fg' Wg' (nth_error_In _ _ HLg'))).
      * intros y Hy. apply (proj2 (Hreturns wl Hwl) r y Hr).
        exact (proj1 (chain_end_last _ _ wl y Hlast) Hy).
      * apply fields_from_intro. intros i x Hx g kk w W p Hpos Hw HW Hp.
        exact (HA i x g kk w W p Hx Hpos Hw HW Hp).
    + split; [ | split; [ exact Hcl_pa | exact Hall_pa ] ].
      exists id, slots, c. auto.
    + unfold partial_or_over_application in Hpoo.
      apply andb_true_iff in Hpoo as [_ Hpoo].
      destruct (Nat.ltb_spec (nparams c) (length A))
        as [Hlt | _]; [ lia | ].
      apply andb_true_iff in Hpoo as [_ Hpoo].
      apply is_one_param_of_kind_value_eq in Hpoo. rewrite Hpoo. reflexivity.
Qed.

Lemma singleton_params_length (a : arity) :
  singleton_params a = true -> length (unarize a) = length a.
Proof.
  unfold unarize. induction a as [ | [k | components] a IH]; simpl;
    [ reflexivity | intros H; f_equal; exact (IH H) | discriminate ].
Qed.

Lemma firstn_length_app {A : Type} (l1 l2 : list A) :
  firstn (length l1) (l1 ++ l2) = l1.
Proof. induction l1 as [ | x l1 IH]; simpl; [ reflexivity | f_equal; exact IH ].
Qed.

Lemma skipn_length_app {A : Type} (l1 l2 : list A) :
  skipn (length l1) (l1 ++ l2) = l2.
Proof. induction l1 as [ | x l1 IH]; simpl; [ reflexivity | exact IH ]. Qed.

(* The fields of a block of an environment with partial applications as
   the run keeps them. *)
Lemma block_fields_pa_ok (statics : static_env) (v : value) (tag : Z)
    (fields : list value) :
  pa_ok statics v -> statics_pa_ok statics ->
  block_of statics v = Some (tag, fields) -> every (pa_ok statics) fields.
Proof.
  intros Hv Hst Hb. destruct v; simpl in Hb, Hv; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:E;
      [ | discriminate ].
    specialize (Hst s allocated E).
    destruct allocated; try discriminate. injection Hb as <- <-. exact Hst.
  - injection Hb as <- <-. exact Hv.
Qed.

(* The fields of a tuple: those of a block, or none. *)
Lemma tuple_fields_block (statics : static_env) (size : nat) (tuple : value)
    (fields : list value) :
  tuple_fields statics size tuple = Some fields ->
  length fields = size
  ∧ (fields = [] ∨ block_of statics tuple = Some (0%Z, fields)).
Proof.
  intros H. destruct size as [ | size].
  - injection H as <-. split; [ reflexivity | left; reflexivity ].
  - destruct (proj1 (FlambdaTupledProofs.tuple_is_block statics (S size)
                       tuple fields ltac:(discriminate)) H)
      as (Hb & Hlen & _).
    split; [ exact Hlen | right; exact Hb ].
Qed.

(* A call of a closure of tupled code: the code takes the fields of the
   tuple, its first argument; with more arguments, it is an
   over-application, whose result is called with the others. *)
Lemma tupled_call_ok (callee : value) (start : node) (vs : list value)
    (args_arity return_arity : arity) (ws returns : list node) (exn : node)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record)
    (shape : call_shape) (id : code_id) (slots : list (value_slot * value))
    (c : code) :
  indirect_ok ->
  flows_ callee start -> pa_ok statics callee ->
  closure_of statics callee = Some (id, slots) ->
  lookup id (static_code statics) = Some c ->
  code_is_tupled c = true ->
  pending start ws vs args_arity returns exn ->
  every (pa_ok statics) vs ->
  cont_ok return_to returns ->
  fits return_to returns (length (unarize return_arity)) ->
  cont_pa_ok statics return_to ->
  cont_ok exn_to [exn] -> cont_pa_ok statics exn_to ->
  statics_flow_ statics -> codes_ok statics -> statics_pa_ok statics ->
  shape_of_closure_call statics callee Indirect_unknown_arity vs args_arity
    return_arity
  = Some shape ->
  result_ok
    (run_call (fun partial => apply_cont_value return_to [partial] statics hp
      rd)
       (Some shape) return_arity return_to exn_to statics hp rd).
Proof.
  intros Hind Hflows Hpa Hclosure Hcode Htupled Hpend Hvs_pa Hret Hfits
    Hret_pa Hexn Hexn_pa Hst Hcodes Hspa Hshape.
  destruct Hpend as (Hlen_ws & Hchain & Hargs & Hreturns & Hraises & Huses).
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (id' & slots' & c' & vs' & args' & Hclosure' & Hcode' & _ & Hafc
        & Hcall).
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  rewrite Hcode in Hcode'. injection Hcode' as <-.
  destruct (code_facts _ _ _ Hcodes Hcode)
    as (Hin_codes & cd & Hcd & _ & Htw & Lp & Lr).
  pose proof (Hind id c Hin_codes) as Hsingle. rewrite Htupled in Hsingle.
  pose proof (singleton_params_length _ Hsingle) as Hsize.
  destruct (Htw Htupled)
    as (W & U & HWs & Hretc & Hexnc & Hcodeid & HparW & Hacc).
  (* The arguments as the code takes them. *)
  unfold arguments_for_code in Hafc. rewrite Htupled in Hafc.
  unfold detuple in Hafc.
  destruct vs as [ | tuple rest]; [ discriminate Hafc | ].
  destruct args_arity as [ | first rest_arity]; [ discriminate Hafc | ].
  destruct (kinds_eqb (map kind_of_value (tuple :: rest))
              (unarize (first :: rest_arity))) eqn:Hk1;
    [ | discriminate Hafc ].
  destruct (kinds_eqb (unarize_component first) [Value]) eqn:Hk2;
    [ | discriminate Hafc ].
  cbn [negb andb] in Hafc.
  destruct (tuple_fields statics (cardinal_unarized (code_params_arity c))
              tuple) as [fields | ] eqn:Hfields; [ | discriminate Hafc ].
  injection Hafc as <- <-.
  apply kinds_eqb_eq in Hk1. apply kinds_eqb_eq in Hk2.
  destruct (tuple_fields_block _ _ _ _ Hfields) as [Hlen_f Hblock].
  unfold cardinal_unarized in Hlen_f. rewrite Hsize in Hlen_f.
  assert (Hlen_rest : length rest = length (unarize rest_arity)).
  { assert (H := f_equal (@length flambda_kind) Hk1). simpl in H.
    rewrite length_map in H. unfold unarize in H. simpl in H.
    rewrite length_app, Hk2 in H. simpl in H. fold (unarize rest_arity) in H.
    lia. }
  assert (Hsizes : arity_sizes (first :: rest_arity)
                   = 1 :: arity_sizes rest_arity).
  { unfold arity_sizes, unarize_per_parameter. simpl. rewrite Hk2.
    reflexivity. }
  destruct ws as [ | w1 ws']; [ discriminate Hlen_ws | ].
  simpl in Hlen_ws. injection Hlen_ws as Hlen_ws.
  (* The witness of the code. *)
  destruct (Hcds id cd Hcd) as (_ & _ & _ & _ & Has_id).
  assert (HW_self : may_come_from sol W W).
  { apply (alloc_self _ _ _ Hcodeid).
    exact (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Has_id)). }
  destruct (closure_alloc _ _ _ _ _ Hst Hflows Hclosure)
    as (b & Hb & Hfb & Hcalled_b).
  unfold called_by, closure_dep_facts in Hcalled_b. simpl in Hcalled_b.
  rewrite Hcd in Hcalled_b. destruct Hcalled_b as (Hprop & _ & Hcw).
  rewrite HWs in Hcw.
  assert (HW : may_come_from sol W w1).
  { exact (proj1 Hchain b W Hb (Hcw W eq_refl) HW_self). }
  assert (Hlive : live (Some id)).
  { exact (Huses 0 w1 W id eq_refl HW Hcodeid). }
  (* The tuple is at [untuple_var], and its fields at the parameters. *)
  assert (Htuple : flows_ tuple U).
  { apply (Hargs 0 tuple 0 0 w1 W U eq_refl); [ | reflexivity | exact HW
                                              | exact HparW ].
    rewrite Hsizes. reflexivity. }
  assert (Hparams : ∀ (i : nat) (x : value) (p : node),
            nth_error fields i = Some x ->
            nth_error (cd_params cd) i = Some p ->
            flows_ x p).
  { intros i x p Hx Hp. destruct Hblock as [-> | Hblock];
      [ destruct i; discriminate Hx | ].
    destruct (block_of_flows _ _ _ _ _ Hst Htuple Hblock)
      as (a & Ha & _ & _ & Hfs).
    destruct (fields_from_elim _ 0 _ i x Hfs Hx) as (from & Hcons & Hfrom).
    refine (flows_read _ _ _ _ _ _ Hc a U p from _ x _ Hcons (Hacc i p Hp) Ha
              Hfrom).
    not_local_field. }
  assert (Hfields_pa : every (pa_ok statics) fields).
  { destruct Hblock as [-> | Hblock]; [ exact I | ].
    exact (block_fields_pa_ok _ _ _ _ (proj1 Hvs_pa) Hspa Hblock). }
  assert (Hm_closure : flows_ callee (cd_my_closure cd)).
  { apply (flows_alias _ _ _ _ _ _ Hc callee _ b
             (alias_propagate _ _ _ Hc _ _ _ Hlive Hprop)).
    exact Hfb. }
  assert (Hexn_code : cont_ok exn_to [cd_exn cd]).
  { apply (cont_ok_compose exn_to [exn] [cd_exn cd] Hexn);
      [ | left; reflexivity ].
    intros [ | i] n' n Hn' Hn v Hv; [ | destruct i; discriminate Hn' ].
    injection Hn' as <-. injection Hn as <-.
    exact (Hraises 0 w1 W _ v eq_refl HW Hexnc Hv). }
  unfold cardinal_unarized in Hcall. rewrite Hsize, <- Hlen_f in Hcall.
  set (args' := arity_of_kinds (repeat Value (length fields)) ++ rest_arity)
    in *.
  assert (Hlen_args : length args' = length fields + length rest_arity).
  { unfold args', arity_of_kinds. rewrite length_app, length_map,
      repeat_length. reflexivity. }
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[Hfull ->] | [(Hover & Hkinds & Hpoo & ->)
                     | (_ & _ & _ & Hnt & _)]];
    [ | | rewrite Htupled in Hnt; discriminate Hnt ].
  - (* A full call: the tuple was the only argument. *)
    assert (rest_arity = []) as ->.
    { destruct rest_arity; [ reflexivity | ].
      rewrite Hlen_args, <- Hlen_f in Hfull. simpl in Hfull. lia. }
    destruct rest as [ | v0 rest]; [ | discriminate Hlen_rest ].
    destruct ws' as [ | w2 ws'']; [ | discriminate Hlen_ws ].
    rewrite app_nil_r.
    cbn [run_call].
    destruct (call_code callee c fields args' return_arity return_to exn_to
                statics hp rd) as [s' | k' vs'' hp'' rd'' | ] eqn:Hcc;
      [ | exfalso; exact (call_code_not_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcc)
      | exact I ].
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ _ Hcc)
      as (callee_vars & _ & _ & _ & Hk3 & Hbind & ->).
    apply (enter_code_ok id c callee fields callee_vars return_to exn_to
             statics hp rd Hcode Hlive); try assumption.
    + first [ reflexivity
            | rewrite Hsize; first [ reflexivity | exact Hlen_f ] ].
    + intros cd' Hcd'. rewrite Hcd in Hcd'. injection Hcd' as <-.
      split; [ exact Hparams | split; [ exact Hm_closure | split;
        [ | exact Hexn_code ] ] ].
      apply (cont_ok_compose return_to returns (cd_return cd) Hret).
      * intros i r' r Hr' Hr v Hv.
        exact (proj1 (Hreturns w1 eq_refl) i r W r' v Hr HW (Hretc i r' Hr')
                 Hv).
      * destruct Hfits as [Hf | Hf]; [ left | right; exact Hf ].
        rewrite Lr, <- Hk3, Hf. reflexivity.
  - (* An over-application: what the code returns is called with the
       arguments after the tuple. *)
    cbn [run_call].
    unfold cardinal_unarized in *. rewrite Hsize, <- Hlen_f in *.
    rewrite firstn_length_app, skipn_length_app.
    assert (Hskip_args : partially_apply args' (length fields) = rest_arity).
    { unfold partially_apply, args', arity_of_kinds.
      replace (length fields)
        with (length (map Singleton (repeat Value (length fields)))) at 1
        by (rewrite length_map, repeat_length; reflexivity).
      apply skipn_length_app. }
    rewrite Hskip_args.
    destruct (call_code callee c fields (firstn (length fields) args')
                (code_result_arity c)
                (Over_application rest rest_arity return_arity return_to
                   exn_to)
                exn_to statics hp rd) as [s' | k' vs'' hp'' rd'' | ] eqn:Hcc;
      [ | exfalso; exact (call_code_not_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcc)
      | exact I ].
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ _ Hcc)
      as (callee_vars & _ & _ & _ & _ & Hbind & ->).
    (* The code returns one value. *)
    unfold partial_or_over_application in Hpoo.
    apply andb_true_iff in Hpoo as [_ Hpoo].
    destruct (Nat.ltb_spec (length (code_params_arity c)) (length args'))
      as [_ | Hge]; [ | lia ].
    apply is_one_param_of_kind_value_eq in Hpoo.
    rewrite Hpoo in Lr. simpl in Lr.
    destruct (cd_return cd) as [ | R0 [ | R1 rr]] eqn:Ecr;
      try discriminate Lr.
    apply (enter_code_ok id c callee fields callee_vars
             (Over_application rest rest_arity return_arity return_to exn_to)
             exn_to statics hp rd Hcode Hlive); try assumption.
    + first [ reflexivity
            | rewrite Hsize; first [ reflexivity | exact Hlen_f ] ].
    + intros cd' Hcd'. rewrite Hcd in Hcd'. injection Hcd' as <-.
      split; [ exact Hparams | split; [ exact Hm_closure | split;
        [ | exact Hexn_code ] ] ].
      rewrite Ecr.
      apply (Over_ok rest rest_arity return_arity return_to exn_to R0 R0 []
               ws' returns exn Hind);
        [ intros v Hv; exact Hv | | exact Hret | exact Hfits | exact Hexn ].
      split; [ exact Hlen_ws | split; [ | split; [ | split; [ | split ] ] ] ].
      * destruct Hchain as [_ Hchain'].
        refine (call_chain_mono _ _ _ _ Hchain').
        intros y Hy. exists W, R0.
        split; [ exact HW | split; [ exact (Hretc 0 R0 eq_refl) | exact Hy ] ].
      * pose proof (args_at_skipn _ _ _ 1 Hargs) as H.
        rewrite Hsizes in H. exact H.
      * intros wl Hwl. apply Hreturns.
        rewrite <- (nth_error_skipn_last (w1 :: ws') 1) by (simpl; lia).
        exact Hwl.
      * intros g w W' E v Hw HW' HE Hv.
        exact (Hraises (S g) w W' E v Hw HW' HE Hv).
      * intros g w W' id' Hw HW' Hcodeid'.
        exact (Huses (S g) w W' id' Hw HW' Hcodeid').
    + apply Over_pa; [ exact (proj2 Hvs_pa) | exact Hret_pa | exact Hexn_pa ].
Qed.

(* A call of no argument of a closure of code that has parameters goes
   wrong. *)
Lemma no_arg_call (statics : static_env) (callee : value) (vs : list value)
    (return_arity : arity) (shape : call_shape) :
  indirect_ok -> codes_ok statics ->
  shape_of_closure_call statics callee Indirect_unknown_arity vs [] return_arity
  = Some shape ->
  False.
Proof.
  intros Hind Hcodes Hshape.
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (id & slots & c & vs' & args_arity' & _ & Hcode & _ & Hafc & Hcall).
  destruct (Hcodes id c Hcode) as [Hin _].
  pose proof (Hind id c Hin) as Hne.
  destruct (code_is_tupled c) eqn:Htupled.
  { unfold arguments_for_code in Hafc. rewrite Htupled in Hafc.
    unfold detuple in Hafc. destruct vs; discriminate Hafc. }
  rewrite (arguments_for_code_not_tupled _ _ _ _ Htupled) in Hafc.
  injection Hafc as <- <-.
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[Hfull _] | [(Hover & _) | (_ & _ & Hpoo & _)]].
  - simpl in Hfull. destruct (code_params_arity c);
      [ contradiction | discriminate ].
  - simpl in Hover. lia.
  - unfold partial_or_over_application in Hpoo.
    apply andb_true_iff in Hpoo as [_ Hpoo]. simpl in Hpoo. discriminate.
Qed.

(* A call of unknown arity, of a closure or of a partial application. *)
Lemma call_ok (callee : value) (start : node) (vs : list value)
    (args_arity return_arity : arity) (ws returns : list node) (exn : node)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record) :
  indirect_ok ->
  flows_ callee start -> pa_ok statics callee ->
  (args_arity <> [] -> pending start ws vs args_arity returns exn) ->
  every (pa_ok statics) vs ->
  cont_ok return_to returns ->
  fits return_to returns (length (unarize return_arity)) ->
  cont_pa_ok statics return_to ->
  cont_ok exn_to [exn] -> cont_pa_ok statics exn_to ->
  statics_flow_ statics -> codes_ok statics -> statics_pa_ok statics ->
  (∀ v : value,
     (∀ r : node, nth_error returns 0 = Some r -> flows_ v r) ->
     pa_ok statics v -> length (unarize return_arity) = 1 ->
     result_ok (apply_cont_value return_to [v] statics hp rd)) ->
  result_ok
    (call_function callee Indirect_unknown_arity vs args_arity return_arity
       return_to exn_to statics hp rd).
Proof.
  intros Hind Hflows Hpa Hpend Hvs_pa Hret Hfits Hret_pa Hexn Hexn_pa Hst
    Hcodes Hspa Hpartial.
  unfold call_function.
  destruct (shape_of_call statics callee Indirect_unknown_arity vs args_arity
              return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  unfold shape_of_call in Hshape.
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ] eqn:Hpa_of.
  - destruct args_arity as [ | a0 args_arity']; [ discriminate Hshape | ].
    destruct (kinds_eqb (map kind_of_value vs) (unarize (a0 :: args_arity')))
      eqn:Hk; [ | discriminate Hshape ].
    destruct callee; try discriminate Hpa_of.
    injection Hpa_of as -> -> ->.
    destruct Hflows
      as (m & ws_p & Hm & Hlen_p & Hchain_p & _ & Hend_p & Hargs_ff).
    destruct Hpa as (Hun & Hcl_pa & Happ_pa).
    apply (closure_call_ok closure m start applied vs applied_arity
             (a0 :: args_arity') return_arity ws_p ws returns exn return_to
             exn_to statics hp rd shape Hind Hm Hcl_pa Hlen_p Hchain_p);
      try assumption.
    + intros i x g kk w W p Hx Hpos Hw HW Hp.
      exact (fields_from_elim _ 0 _ i x Hargs_ff Hx g kk w W p Hpos Hw HW Hp).
    + intros id sl c Hof Hcode.
      destruct Hun as (id' & sl' & c' & Hof' & Hcode' & Hlt & Hnt).
      rewrite Hof in Hof'. injection Hof' as <- <-.
      rewrite Hcode in Hcode'. injection Hcode' as <-. auto.
    + discriminate.
    + apply Hpend. discriminate.
    + right. apply kinds_eqb_eq in Hk. rewrite <- Hk, length_map.
      reflexivity.
  - destruct args_arity as [ | a0 args_arity'].
    + exfalso. exact (no_arg_call _ _ _ _ _ Hind Hcodes Hshape).
    + destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
        as (id0 & slots0 & c0 & vs0 & a0' & Hclosure0 & Hcode0 & _ & _ & _).
      destruct (code_is_tupled c0) eqn:Htupled.
      { exact (tupled_call_ok callee start vs (a0 :: args_arity') return_arity
                 ws returns exn return_to exn_to statics hp rd shape id0 slots0
                   c0
                 Hind Hflows Hpa Hclosure0 Hcode0 Htupled
                 (Hpend ltac:(discriminate)) Hvs_pa Hret Hfits Hret_pa Hexn
                 Hexn_pa Hst Hcodes Hspa Hshape). }
      apply (closure_call_ok callee start start [] vs [] (a0 :: args_arity')
               return_arity [] ws returns exn return_to exn_to statics hp rd
                 shape
               Hind Hflows Hpa eq_refl I); try assumption.
      * intros i x g kk w W p Hx. destruct i; discriminate Hx.
      * intros y Hy. exact Hy.
      * intros id sl c Hof Hcode.
        rewrite Hclosure0 in Hof. injection Hof as <- <-.
        rewrite Hcode0 in Hcode. injection Hcode as <-.
        destruct (Hcodes id0 c0 Hcode0) as [Hin _].
        pose proof (Hind id0 c0 Hin) as Hne. rewrite Htupled in Hne.
        split; [ | exact Htupled ].
        destruct (code_params_arity c0); [ contradiction | simpl; lia ].
      * discriminate.
      * apply Hpend. discriminate.
      * left. reflexivity.
      * exact I.
Qed.

(* What a continuation is, as [cont_ok] has it. *)
Lemma cont_ok_outer_inv (k : continuation) (ns : list node) :
  cont_ok (Outer k) ns -> transfers ns (unit_cont_nodes u k).
Proof. intros H. inversion H; subst. assumption. Qed.

Lemma cont_ok_closure_inv (h : continuation_handler) (vars : env)
    (K0 : cont_env) (recursive : list (continuation * continuation_handler))
    (ns : list node) :
  cont_ok (Closure h vars K0 recursive) ns ->
  ∃ (ctx : option code_id) (ce0 : finite_map (list node)),
    env_flows_ vars ∧ conts_ok ctx K0 ce0 ∧ live ctx
    ∧ (∀ (k' : continuation) (h' : continuation_handler),
         (k', h') ∈ recursive ->
         handler_edges_ ctx (rec_conts ce0 recursive) h'
         ∧ handler_counted ctx (rec_conts ce0 recursive) h')
    ∧ handler_edges_ ctx (rec_conts ce0 recursive) h
    ∧ handler_counted ctx (rec_conts ce0 recursive) h
    ∧ transfers ns (handler_params h).
Proof.
  intros H. inversion H; subst. do 2 eexists.
  repeat (split; [ eassumption | ]). eassumption.
Qed.

Lemma cont_ok_extra_inv (c : cont_value) (extra : list value)
    (ns : list node) :
  cont_ok (Extra_args c extra) ns ->
  ∃ ns' : list node,
    cont_ok c (ns ++ ns')
    ∧ fits c (ns ++ ns') (length ns + length extra)
    ∧ ∀ (i : nat) (v : value) (n : node),
        nth_error extra i = Some v -> nth_error ns' i = Some n -> flows_ v n.
Proof.
  intros H. inversion H; subst. eexists.
  split; [ eassumption | split; [ eassumption | eassumption ] ].
Qed.

Lemma cont_ok_over_inv (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value)
    (ns : list node) :
  cont_ok
    (Over_application remaining remaining_arity return_arity return_to exn_to)
    ns ->
  ∃ (n n0 : node) (ns0 ws returns : list node) (exn : node),
    ns = n :: ns0
    ∧ indirect_ok
    ∧ (∀ v : value, flows_ v n -> flows_ v n0)
    ∧ pending n0 ws remaining remaining_arity returns exn
    ∧ cont_ok return_to returns
    ∧ fits return_to returns (length (unarize return_arity))
    ∧ cont_ok exn_to [exn].
Proof.
  intros H. inversion H; subst. do 6 eexists.
  split; [ reflexivity | ].
  repeat (split; [ eassumption | ]). eassumption.
Qed.

Lemma cont_pa_closure_inv (statics : static_env) (h : continuation_handler)
    (vars : env) (conts : cont_env)
    (recursive : list (continuation * continuation_handler)) :
  cont_pa_ok statics (Closure h vars conts recursive) ->
  env_pa_ok statics vars ∧ conts_pa_ok statics conts.
Proof. intros H. inversion H; subst. split; assumption. Qed.

Lemma cont_pa_extra_inv (statics : static_env) (c : cont_value)
    (extra : list value) :
  cont_pa_ok statics (Extra_args c extra) ->
  cont_pa_ok statics c ∧ every (pa_ok statics) extra.
Proof. intros H. inversion H; subst. split; assumption. Qed.

Lemma cont_pa_over_inv (statics : static_env) (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_pa_ok statics
    (Over_application remaining remaining_arity return_arity return_to
       exn_to) ->
  every (pa_ok statics) remaining
  ∧ cont_pa_ok statics return_to ∧ cont_pa_ok statics exn_to.
Proof. intros H. inversion H; subst. auto. Qed.

(* A continuation given values at its nodes. *)
Lemma apply_ok (c : cont_value) :
  ∀ (ns : list node) (vs : list value) (statics : static_env) (hp : heap) (rd :
    read_record),
    cont_ok c ns -> cont_pa_ok statics c ->
    (∀ (i : nat) (v : value) (n : node),
       nth_error vs i = Some v -> nth_error ns i = Some n -> flows_ v n) ->
    every (pa_ok statics) vs ->
    fits c ns (length vs) ->
    statics_flow_ statics -> codes_ok statics -> statics_pa_ok statics ->
    result_ok (apply_cont_value c vs statics hp rd).
Proof.
  induction c as [h vars K0 recursive | k | c IH extra
                 | remaining remaining_arity return_arity return_to exn_to
                     IHret IHexn]
    using cont_value_over_ind;
    intros ns vs statics hp rd Hok Hpa Hvs Hvs_pa Hfits Hst Hcodes Hspa.
  - (* A handler. *)
    destruct (cont_ok_closure_inv _ _ _ _ _ Hok)
      as (ctx0 & ce0 & Hvars0 & Hconts0 & Hlive0 & Hrec & Hh & Hcount0 & Ht).
    destruct (cont_pa_closure_inv _ _ _ _ _ Hpa) as [Hvars_pa HK0_pa].
    destruct h as [params metadata handler]. cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [handler_vars | ] eqn:Hbind;
      [ | exact I ].
    pose proof (bind_params_length _ _ _ _ Hbind) as Hlen_params.
    exists ctx0, (rec_conts ce0 recursive). simpl_state.
    split; [ exact Hh | split; [ exact Hlive0 | split; [ exact Hcount0 | ] ] ].
    split; [ | split; [ | split; [ exact Hst | split; [ exact Hcodes | ] ] ] ].
    + intros y w Hy.
      destruct (bind_params_lookup _ _ _ _ _ _ Hbind Hy)
        as [Hold | (i & Hy_nth & Hw_nth)]; [ exact (Hvars0 y w Hold) | ].
      destruct (nth_error ns i) as [n | ] eqn:Hn.
      * apply (Ht i w n (Node_variable y) Hn);
          [ | exact (Hvs i w n Hw_nth Hn) ].
        simpl. apply map_nth_error. exact Hy_nth.
      * exfalso. apply nth_error_None in Hn.
        assert (i < length vs) by (apply nth_error_Some; congruence).
        destruct Hfits as [Hf | (t & Ht' & Hle)]; [ lia | ].
        simpl in Ht'. injection Ht' as <-. simpl in Hle.
        rewrite length_map in Hle. lia.
    + apply conts_ok_rec; assumption.
    + split; [ | split; [ exact Hspa | ] ].
      * intros y w Hy.
        destruct (bind_params_lookup _ _ _ _ _ _ Hbind Hy)
          as [Hold | (i & _ & Hw_nth)];
          [ exact (Hvars_pa y w Hold)
          | exact (proj1 (every_in _ _ _) Hvs_pa w (nth_error_In _ _ Hw_nth))
          ].
      * intros k' c' Hin. apply in_app_or in Hin as [Hin | Hin];
          [ | exact (HK0_pa k' c' Hin) ].
        unfold rec_closures in Hin.
        apply in_map_iff in Hin as ([k'' h''] & [= <- <-] & _).
        apply Closure_pa; [ exact Hvars_pa | exact HK0_pa ].
  - (* A way out. *)
    pose proof (cont_ok_outer_inv _ _ Hok) as Ht.
    simpl. intros i v p Hv Hp.
    destruct (nth_error ns i) as [n | ] eqn:Hn.
    + exact (Ht i v n p Hn Hp (Hvs i v n Hv Hn)).
    + exfalso. apply nth_error_None in Hn.
      assert (i < length vs) by (apply nth_error_Some; congruence).
      assert (i < length (unit_cont_nodes u k))
        by (apply nth_error_Some; congruence).
      destruct Hfits as [Hf | (t & Ht' & Hle)]; [ lia | ].
      simpl in Ht'. injection Ht' as <-. lia.
  - (* Extra arguments. *)
    destruct (cont_ok_extra_inv _ _ _ Hok) as (ns' & Hc' & Hfits' & Hextra).
    destruct (cont_pa_extra_inv _ _ _ Hpa) as [Hc_pa Hextra_pa].
    destruct Hfits as [Hf | (t & Ht & _)]; [ | discriminate Ht ].
    cbn [apply_cont_value].
    apply (IH (ns ++ ns')); try assumption.
    + intros i v n Hv Hn.
      destruct (Nat.ltb_spec i (length vs)) as [Hi | Hi].
      * rewrite nth_error_app1 in Hv by exact Hi.
        rewrite nth_error_app1 in Hn by lia. exact (Hvs i v n Hv Hn).
      * rewrite nth_error_app2 in Hv by exact Hi.
        rewrite nth_error_app2 in Hn by lia. rewrite Hf in Hv.
        exact (Hextra _ v n Hv Hn).
    + exact (every_app_intro _ _ _ Hvs_pa Hextra_pa).
    + rewrite length_app, Hf. exact Hfits'.
  - (* What is left of an over-application. *)
    destruct (cont_ok_over_inv _ _ _ _ _ _ Hok)
      as (n & n0 & ns0 & ws & returns & exn & -> & Hind & Htr & Hpend & Hret
          & Hfits_ret & Hexn).
    destruct (cont_pa_over_inv _ _ _ _ _ _ Hpa)
      as (Hrem_pa & Hret_pa & Hexn_pa).
    rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs']]; [ exact I | | exact I ].
    apply (call_ok v n0 remaining remaining_arity return_arity ws returns exn
             return_to exn_to statics hp rd Hind); try assumption.
    + apply Htr. exact (Hvs 0 v n eq_refl eq_refl).
    + exact (proj1 Hvs_pa).
    + intros _. exact Hpend.
    + intros v'' Hv'' Hv''_pa Hone.
      apply (IHret returns [v''] statics hp rd Hret Hret_pa); try assumption.
      * intros [ | i] x r Hx Hr; [ injection Hx as <-; exact (Hv'' r Hr)
                                 | destruct i; discriminate Hx ].
      * split; [ exact Hv''_pa | exact I ].
      * destruct Hfits_ret as [Hf | Hf]; [ left | right; exact Hf ].
        rewrite <- Hf, Hone. reflexivity.
Qed.

Lemma nth_error_combine_eq {A B : Type} (l1 : list A) (l2 : list B)
    (g : nat) (a : A) (b : B) :
  nth_error l1 g = Some a -> nth_error l2 g = Some b ->
  nth_error (combine l1 l2) g = Some (a, b).
Proof.
  revert l2 g. induction l1 as [ | x l1 IH]; intros [ | y l2] [ | g] Ha Hb;
    simpl in *; try discriminate; [ congruence | exact (IH _ _ Ha Hb) ].
Qed.

(* The witnesses of a call of unknown arity read, one from the other. *)
Lemma caller_call_chain (returns : list node) (exn called_node : node) :
  ∀ (L : list (list simple)) (ws : list node) (P : node -> Prop),
    caller_chain_edges G returns exn called_node (combine L ws) ->
    length ws = length L ->
    (∀ w : node, hd_error ws = Some w -> called_from G sol P w) ->
    call_chain G sol P ws.
Proof.
  intros L ws. revert L.
  induction ws as [ | w ws IH]; intros L P Hch Hlen Hfirst; [ exact I | ].
  destruct L as [ | grp L]; [ discriminate Hlen | ].
  simpl in Hch, Hlen. destruct Hch as (_ & _ & _ & Hnext & Hrest).
  split; [ exact (Hfirst w eq_refl) | ].
  apply (IH L); [ exact Hrest | lia | ].
  intros w' Hw'. destruct ws as [ | w'' ws']; [ discriminate Hw' | ].
  simpl in Hw'. injection Hw' as ->.
  destruct L as [ | grp' L']; [ discriminate Hlen | ].
  simpl in Hnext. destruct Hnext as (v & Hv & Hw'v).
  intros y W (W0 & R & HW0 & HR & Hy) HyW HW.
  assert (Hyv : may_come_from sol y v).
  { refine (may_come_from_read _ _ _ Hc y W0 w v R _ _ HR Hv HW0 Hy).
    not_local_field. }
  refine (may_come_from_read _ _ _ Hc W y v w' W _ _ HyW Hw'v Hyv HW).
  not_local_field.
Qed.

(* The call of unknown arity that a term makes. *)
Lemma indirect_pending (ctx : option code_id) (ap : apply_expr) (vars : env)
    (vs : list value) (returns : list node) (exn : node) (ws : list node)
    (apply_node called_node : node) :
  live ctx -> env_flows_ vars -> simples_noted noted (ap_args ap) ->
  eval_simples vars (ap_args ap) = Some vs ->
  ap_args_arity ap <> [] ->
  length ws = length (call_groups ap) ->
  caller_chain_edges G returns exn called_node (combine (call_groups ap) ws) ->
  g_any_usage G called_node ->
  (∀ w : node, hd_error ws = Some w -> cond_alias_edge G ctx w apply_node) ->
  g_accessor G apply_node (Field_call_witness Unknown_arity_code_pointer)
    (node_of_simple (ap_callee ap)) ->
  pending (node_of_simple (ap_callee ap)) ws vs (ap_args_arity ap) returns
    exn.
Proof.
  intros Hlive Hvars Hnoted Hvs Hne Hlen Hch Hused Hcond Happly.
  unfold call_groups in Hlen, Hch.
  rewrite (group_by_parameter_groups _ _ Hne) in Hlen, Hch.
  set (groups := group_by_sizes (arity_sizes (ap_args_arity ap)) (ap_args ap))
    in *.
  assert (Hlen' : length ws = length (ap_args_arity ap)).
  { rewrite Hlen. unfold groups.
    rewrite length_group_by_sizes, length_arity_sizes. reflexivity. }
  assert (Hnth : ∀ (g : nat) (w : node),
            nth_error ws g = Some w ->
            ∃ grp : list simple,
              nth_error (combine groups ws) g = Some (grp, w)).
  { intros g w Hw. destruct (nth_error groups g) as [grp | ] eqn:Hg.
    - exists grp. exact (nth_error_combine_eq _ _ _ _ _ Hg Hw).
    - apply nth_error_None in Hg.
      assert (g < length ws) by (apply nth_error_Some; congruence). lia. }
  split; [ exact Hlen' | split; [ | split; [ | split; [ | split ] ] ] ].
  - apply (caller_call_chain returns exn called_node groups ws _ Hch Hlen).
    intros w Hw y W Hy HyW HW.
    destruct ws as [ | w1 ws']; [ discriminate Hw | ].
    simpl in Hw. injection Hw as <-.
    apply (may_come_from_alias _ _ _ Hc _ _ apply_node);
      [ exact (cond_alias_alias _ _ _ Hlive (Hcond w1 eq_refl)) | ].
    refine (may_come_from_read _ _ _ Hc W y (node_of_simple (ap_callee ap))
              apply_node W _ _ HyW Happly Hy HW).
    not_local_field.
  - intros i x g kk w W p Hx Hpos Hw HW Hp.
    destruct (eval_simples_nth _ _ _ _ _ Hvs Hx) as (s & Hs & Hsx).
    destruct (group_by_sizes_nth _ _ _ _ _ _ Hpos Hs) as (grp & Hgrp & Hs').
    fold groups in Hgrp.
    pose proof (nth_error_combine_eq _ _ _ _ _ Hgrp Hw) as Hgw.
    destruct (caller_chain_nth _ _ _ _ _ _ _ Hch Hgw) as (Hargs & _).
    pose proof (simple_flows _ _ _ Hvars (Hnoted s (nth_error_In _ _ Hs)) Hsx)
      as Hfs.
    refine (argument_transfer _ _ _ Hc (flows_ x) W w p (node_of_simple s) kk
              _ _ HW (Hargs kk s Hs') Hp (flows_hs _ _ _ _ _ _ Hc _ _ Hfs) Hfs).
    + intros x' y'. exact (flows_alias _ _ _ _ _ _ Hc x x' y').
    + intros x' m. exact (flows_escape _ _ _ _ _ _ Hc x x' m).
  - intros w Hw.
    destruct (Hnth _ _ Hw) as (grp & Hgw).
    destruct (caller_chain_nth _ _ _ _ _ _ _ Hch Hgw)
      as (_ & _ & _ & Hlast & _).
    assert (HS : S (pred (length ws)) = length (combine groups ws)).
    { rewrite length_combine, <- Hlen.
      destruct ws; [ discriminate Hw | simpl; lia ]. }
    specialize (Hlast HS). split.
    + intros i r W R v Hr HW HR Hv.
      refine (flows_read _ _ _ _ _ _ Hc W w r R _ v _ HR (Hlast i r Hr) HW Hv).
      not_local_field.
    + intros r y Hr (W & R & HW & HR & Hy).
      refine (may_come_from_read _ _ _ Hc y W w r R _ _ HR (Hlast 0 r Hr) HW
                Hy).
      not_local_field.
  - intros g w W E v Hw HW HE Hv.
    destruct (Hnth _ _ Hw) as (grp & Hgw).
    destruct (caller_chain_nth _ _ _ _ _ _ _ Hch Hgw) as (_ & Hexnacc & _).
    refine (flows_read _ _ _ _ _ _ Hc W w exn E _ v _ HE Hexnacc HW Hv).
    not_local_field.
  - intros g w W id Hw HW Hcodeid.
    destruct (Hnth _ _ Hw) as (grp & Hgw).
    destruct (caller_chain_nth _ _ _ _ _ _ _ Hch Hgw)
      as (_ & _ & Hcalled & _).
    refine (read_any_usage _ _ _ Hc W w called_node _ _ _ Hcodeid Hcalled HW
              (any_usage_input _ _ _ Hc _ Hused)).
    not_local_field.
Qed.

(* The continuation that a name stands for, with nodes that agree with
   those of [ce]. *)
Lemma resolve_ok (ctx : option code_id) (ce : finite_map (list node))
    (K : cont_env) (statics : static_env) (k : continuation) (n : nat)
    (ns : list node) :
  conts_ok ctx K ce -> conts_pa_ok statics K ->
  (ctx <> None -> count_ok (counts ce) k n = true) ->
  (∀ ns' : list node, lookup k ce = Some ns' -> ns = ns') ->
  cont_ok (resolve k K) ns
  ∧ fits (resolve k K) ns n
  ∧ cont_pa_ok statics (resolve k K).
Proof.
  intros Hconts HK_pa Hcount Hns. unfold resolve.
  destruct (lookup k K) as [c | ] eqn:HK.
  - destruct (conts_ok_lookup _ _ _ _ _ Hconts HK) as (ns0 & Hns0 & Hok & Htop).
    rewrite <- (Hns ns0 Hns0) in Hok, Htop.
    split; [ exact Hok | split; [ | exact (HK_pa k c (lookup_in _ _ _ HK)) ] ].
    destruct ctx as [id | ].
    + left. specialize (Hcount ltac:(discriminate)).
      destruct (count_ok_lookup _ _ _ Hcount) as (ns' & Hns' & Hl).
      rewrite Hns0 in Hns'. injection Hns' as <-.
      rewrite (Hns ns0 Hns0). lia.
    + right. exists (length ns). split; [ exact (Htop eq_refl) | lia ].
  - destruct ctx as [id | ].
    + exfalso. pose proof (conts_ok_none _ _ _ _ Hconts HK) as Hnone.
      specialize (Hcount ltac:(discriminate)).
      destruct (count_ok_lookup _ _ _ Hcount) as (ns' & Hns' & _).
      simpl in Hnone. congruence.
    + pose proof (conts_ok_none _ _ _ _ Hconts HK) as Hnone.
      change (lookup k ce = lookup k (unit_conts u)) in Hnone.
      split; [ | split; [ | apply Outer_pa ] ].
      * apply Outer_ok. unfold unit_cont_nodes.
        destruct (lookup k (unit_conts u)) as [l | ] eqn:Hu.
        -- rewrite (Hns l Hnone).
           intros i v n0 p Hn0 Hp Hv. rewrite Hn0 in Hp. injection Hp as <-.
           exact Hv.
        -- intros i v n0 p _ Hp. destruct i; discriminate Hp.
      * right. simpl. unfold unit_cont_nodes.
        destruct (lookup k (unit_conts u)) as [l | ] eqn:Hu.
        -- rewrite (Hns l Hnone).
           exists (length l). split; [ reflexivity | lia ].
        -- exists 0. split; [ reflexivity | lia ].
Qed.

(* The continuations of a call. *)
Lemma apply_conts (ctx : option code_id) (ce : finite_map (list node))
    (ap : apply_expr) (vars : env) (K : cont_env) (statics : static_env)
    (extra : list value) (returns : list node) (exn : node) :
  counted ctx ce (Apply ap) -> env_flows_ vars -> env_pa_ok statics vars ->
  conts_ok ctx K ce -> conts_pa_ok statics K ->
  (∀ params : list node,
     lookup (ap_continuation ap) ce = Some params -> returns = params) ->
  (∀ (p : node) (params : list node),
     lookup (ap_exn_continuation ap) ce = Some (p :: params) ->
     exn = p
     ∧ simples_noted noted (exn_extra_simples ap)
     ∧ ∀ (q : node) (s : simple),
         (q, s) ∈ combine params (exn_extra_simples ap) ->
         g_alias G q (node_of_simple s)) ->
  eval_simples vars (exn_extra_simples ap) = Some extra ->
  (cont_ok (resolve (ap_continuation ap) K) returns
   ∧ fits (resolve (ap_continuation ap) K) returns
       (length (unarize (ap_return_arity ap)))
   ∧ cont_pa_ok statics (resolve (ap_continuation ap) K))
  ∧ cont_ok (with_extra_args (resolve (ap_exn_continuation ap) K) extra) [exn]
  ∧ cont_pa_ok statics
      (with_extra_args (resolve (ap_exn_continuation ap) K) extra).
Proof.
  intros Hcount Hvars Hvars_pa Hconts HK_pa Hret Hexn Hextra.
  assert (Hlen_extra : length extra = length (ap_exn_extra_args ap)).
  { rewrite (eval_simples_length _ _ _ Hextra). unfold exn_extra_simples.
    rewrite length_map. reflexivity. }
  assert (Hcount_ret : ctx <> None ->
            count_ok (counts ce) (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) = true).
  { intros Hctx. specialize (Hcount Hctx). simpl in Hcount.
    apply andb_true_iff in Hcount as [H _]. exact H. }
  assert (Hcount_exn : ctx <> None ->
            count_ok (counts ce) (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) = true).
  { intros Hctx. specialize (Hcount Hctx). simpl in Hcount.
    apply andb_true_iff in Hcount as [_ H]. exact H. }
  split; [ exact (resolve_ok _ _ _ _ _ _ _ Hconts HK_pa Hcount_ret Hret) | ].
  set (c0 := resolve (ap_exn_continuation ap) K).
  assert (Hc0 : ∃ nsx : list node,
            cont_ok c0 (exn :: nsx)
            ∧ fits c0 (exn :: nsx) (S (length extra))
            ∧ (∀ (i : nat) (v : value) (n : node),
                 nth_error extra i = Some v -> nth_error nsx i = Some n ->
                 flows_ v n)
            ∧ cont_pa_ok statics c0).
  { destruct (lookup (ap_exn_continuation ap) ce) as [[ | p params] | ]
      eqn:Hce.
    - destruct (resolve_ok _ _ _ _ _ _ [] Hconts HK_pa Hcount_exn
                  ltac:(intros ns' H; congruence))
        as (Hok & Hfits & Hpa).
      destruct Hfits as [Hf | (t & Ht & Hle)]; [ discriminate Hf | ].
      simpl in Hle. assert (t = 0) as -> by lia.
      exists []. split; [ | split; [ | split; [ | exact Hpa ] ] ].
      + apply (cont_ok_compose c0 [] [exn] Hok);
          [ intros [ | i] n' n _ Hn; discriminate Hn | ].
        right. exists 0. split; [ exact Ht | simpl; lia ].
      + right. exists 0. split; [ exact Ht | simpl; lia ].
      + intros i v n _ Hn. destruct i; discriminate Hn.
    - destruct (Hexn p params eq_refl) as (-> & Hnoted & Haliases).
      destruct (resolve_ok _ _ _ _ _ _ (p :: params) Hconts HK_pa Hcount_exn
                  ltac:(intros ns' H; congruence))
        as (Hok & Hfits & Hpa).
      exists params. split; [ exact Hok | split; [ | split; [ | exact Hpa ] ] ].
      + rewrite Hlen_extra. exact Hfits.
      + intros i v n Hv Hn.
        destruct (eval_simples_nth _ _ _ _ _ Hextra Hv) as (s & Hs & Hsv).
        apply (flows_alias _ _ _ _ _ _ Hc v _ (node_of_simple s)).
        * apply (alias_input _ _ _ Hc). apply Haliases.
          exact (nth_error_combine _ _ _ _ _ Hn Hs).
        * exact (simple_flows _ _ _ Hvars (Hnoted s (nth_error_In _ _ Hs))
                   Hsv).
    - destruct (resolve_ok _ _ _ _ _ _ [exn] Hconts HK_pa Hcount_exn
                  ltac:(intros ns' H; congruence))
        as (Hok & Hfits & Hpa).
      exists []. split; [ exact Hok | split; [ | split; [ | exact Hpa ] ] ].
      + destruct Hfits as [Hf | Hf]; [ left | right; exact Hf ].
        simpl in Hf |- *. rewrite Hlen_extra. exact Hf.
      + intros i v n _ Hn. destruct i; discriminate Hn. }
  destruct Hc0 as (nsx & Hok & Hfits & Hextra_flow & Hpa).
  destruct extra as [ | x extra'].
  - simpl. split; [ | exact Hpa ].
    apply (cont_ok_compose c0 (exn :: nsx) [exn] Hok).
    + intros [ | i] n' n Hn' Hn v Hv; [ | destruct i; discriminate Hn' ].
      simpl in Hn', Hn. congruence.
    + destruct Hfits as [Hf | Hf]; [ left; simpl in *; lia | right; exact Hf ].
  - simpl. split.
    + apply (Extra_ok c0 (x :: extra') [exn] nsx); [ exact Hok | | ].
      * simpl. exact Hfits.
      * exact Hextra_flow.
    + apply Extra_pa; [ exact Hpa | ].
      exact (simples_pa_ok _ _ _ _ Hvars_pa Hextra).
Qed.

(* ------------------------------------------------------------------------ *)
(* Steps                                                                    *)
(* ------------------------------------------------------------------------ *)

(* An [Apply_cont] gives its continuation as many values as it has
   nodes, or, at toplevel, a handler or a way out that has as many nodes
   as it takes values. *)
Lemma apply_cont_fits (ctx : option code_id) (ce : finite_map (list node))
    (ac : apply_cont) (ns : list node) (c : cont_value) :
  counted ctx ce (Apply_cont ac) ->
  lookup (ac_continuation ac) ce = Some ns ->
  (ctx = None -> target_length c = Some (length ns)) ->
  fits c ns (length (ac_args ac)).
Proof.
  intros Hcount Hns Htop. destruct ctx as [id | ].
  - left. specialize (Hcount ltac:(discriminate)). simpl in Hcount.
    destruct (count_ok_lookup _ _ _ Hcount) as (ns' & Hns' & Hlen).
    rewrite Hns in Hns'. injection Hns' as <-. lia.
  - right. exists (length ns). split; [ exact (Htop eq_refl) | lia ].
Qed.

(* ------------------------------------------------------------------------ *)
(* The heap                                                                 *)
(* ------------------------------------------------------------------------ *)

(* What a primitive of the heap allocates. *)
Definition heap_allocates (q : heap_primitive) : bool :=
  match q with
  | Make_block_with_identity _ _ | Make_mutable_array _ => true
  | Block_set _ _ _ | Array_set _ _ => false
  end.

(* The location that the [Let] of a state allocates is given the node of
   its variable. *)
Definition alloc_ok (s : state) : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (q : heap_primitive)
    (args : list simple) (body : expr),
    state_expr s = Let x kind (Prim (Heap q) args) body ->
    heap_allocates q = true ->
    located (fresh_location (state_heap s)) = Some (Node_variable x).

(* The partial applications that the cells hold are as the run makes
   them. *)
Definition heap_pa_ok (statics : static_env) (hp : heap) : Prop :=
  ∀ (l : location) (c : cell) (v : value),
    lookup l hp = Some c -> v ∈ values_of_cell c -> pa_ok statics v.

(* The invariant of the heap of a state. *)
Definition heap_ok (s : state) : Prop :=
  heap_flows_ (state_heap s) ∧ heap_pa_ok (state_statics s) (state_heap s).

Lemma flows_location (l : location) (n : node) :
  flows_ (Value_location l) n ->
  ∃ a : node, located l = Some a ∧ may_come_from sol a n.
Proof.
  simpl. destruct (located l) as [a | ]; [ intros H; exists a; auto | ].
  intros [].
Qed.

(* A location among the values of the arguments is that of a variable. *)
Lemma location_simple (vars : env) (s : simple) (l : location) :
  eval_simple vars s = Some (Value_location l) -> ∃ y : variable, s = Var y.
Proof.
  destruct s as [y | sym | c]; simpl; [ eauto | discriminate | discriminate ].
Qed.

(* The variable of a primitive with arbitrary effects, in code that is
   used, has any usage. *)
Lemma cond_use_live (ctx : option code_id) (n : node) :
  live ctx -> cond_use_edge G ctx n -> au n.
Proof.
  unfold live, cond_use_edge. destruct ctx as [id | ].
  - intros Hlive Huse.
    exact (any_usage_use _ _ _ Hc _ _ (au_hu _ _ _ Hc _ Hlive) Huse).
  - intros _ H. exact (any_usage_input _ _ _ Hc _ H).
Qed.

(* What an argument of a primitive of the default case holds is a
   constant, or a value at a node that has any usage where the variable
   of the primitive has a usage. *)
Lemma simple_used (statics : static_env) (vars : env) (x : variable)
    (args : list simple) (s : simple) (v : value) :
  statics_flow_ statics -> env_flows_ vars -> s ∈ args ->
  (∀ from : node,
     from ∈ free_names_of_simples args -> g_use G (Node_variable x) from) ->
  (∀ sym : symbol,
     s = Symbol sym -> lookup sym (static_values statics) <> None) ->
  eval_simple vars s = Some v ->
  (∃ c : const, v = Value_const c)
  ∨ ∃ m : node, flows_ v m ∧ (hu (Node_variable x) -> au m).
Proof.
  intros Hst Hvars Hin Huse Hbound Hs.
  destruct s as [y | sym | c]; simpl in Hs.
  - right. exists (Node_variable y). split; [ exact (Hvars y v Hs) | ].
    intros Hhu.
    exact (any_usage_use _ _ _ Hc _ _ Hhu (Huse _ (free_names_var _ _ Hin))).
  - injection Hs as <-. right. exists (Node_symbol sym). split.
    + destruct (lookup sym (static_values statics)) as [allocated | ] eqn:Hl;
        [ | destruct (Hbound sym eq_refl Hl) ].
      exact (proj1 (Hst sym allocated Hl)).
    + intros Hhu.
      exact (any_usage_use _ _ _ Hc _ _ Hhu
               (Huse _ (free_names_symbol _ _ Hin))).
  - injection Hs as <-. left. eexists. reflexivity.
Qed.

Lemma simple_escapes (statics : static_env) (vars : env) (x : variable)
    (args : list simple) (s : simple) (v : value) :
  statics_flow_ statics -> env_flows_ vars -> s ∈ args ->
  (∀ from : node,
     from ∈ free_names_of_simples args -> g_use G (Node_variable x) from) ->
  (∀ sym : symbol,
     s = Symbol sym -> lookup sym (static_values statics) <> None) ->
  au (Node_variable x) ->
  eval_simple vars s = Some v ->
  escapes G sol keep called located v.
Proof.
  intros Hst Hvars Hin Huse Hbound Hau Hs.
  destruct (simple_used _ _ _ _ _ _ Hst Hvars Hin Huse Hbound Hs)
    as [Hconst | (m & Hm & Hmu)]; [ left; exact Hconst | ].
  right. exists m. split; [ exact Hm | exact (Hmu (au_hu _ _ _ Hc _ Hau)) ].
Qed.

(* A value that escapes may be at every node with any source. *)
Lemma escapes_flows (v : value) (n : node) :
  escapes G sol keep called located v -> as_ n -> flows_ v n.
Proof.
  intros [(c & ->) | (m & Hm & Hau)] Has; [ exact Has | ].
  exact (flows_escape _ _ _ _ _ _ Hc v n m Has (fun _ => Hau) Hm).
Qed.

Lemma fields_from_replace (P : nat -> value -> Prop) (k i : nat)
    (new : value) (fields fields' : list value) :
  fields_from P k fields -> replace_nth i new fields = Some fields' ->
  (∀ old : value,
     nth_error fields i = Some old -> P (k + i) old -> P (k + i) new) ->
  fields_from P k fields'.
Proof.
  revert k i fields'.
  induction fields as [ | f fields IH]; intros k i fields' Hf Hr Hnew;
    [ destruct i; discriminate Hr | ].
  destruct i as [ | i]; simpl in Hr.
  - injection Hr as <-. destruct Hf as [Hf Hfs]. split; [ | exact Hfs ].
    specialize (Hnew f eq_refl). rewrite Nat.add_0_r in Hnew.
    exact (Hnew Hf).
  - destruct (replace_nth i new fields) as [l' | ] eqn:E; [ | discriminate Hr ].
    injection Hr as <-. destruct Hf as [Hf Hfs]. split; [ exact Hf | ].
    apply (IH (S k) i l' Hfs E). intros old Hold HP.
    replace (S k + i) with (k + S i) in * by lia.
    exact (Hnew old Hold HP).
Qed.

Lemma store_field_replace (kind : block_access_kind) (field : Z)
    (new : value) (tag : Z) (fields fields' : list value) :
  store_field kind field new tag fields = Some fields' ->
  replace_nth (Z.to_nat field) new fields = Some fields'.
Proof.
  unfold store_field. destruct kind.
  match goal with |- (if ?b then _ else _) = _ -> _ => destruct b end;
    [ exact (fun H => H) | discriminate ].
Qed.

Lemma load_field_nth (kind : block_access_kind) (field : Z) (tag : Z)
    (fields : list value) (v : value) :
  load_field kind field tag fields = Some v ->
  nth_error fields (Z.to_nat field) = Some v.
Proof.
  unfold load_field. destruct kind.
  match goal with |- (if ?b then _ else _) = _ -> _ => destruct b end;
    [ | discriminate ].
  destruct (nth_error fields (Z.to_nat field)) as [w | ];
    [ | discriminate ].
  match goal with |- (if ?b then _ else _) = _ -> _ => destruct b end;
    [ | discriminate ].
  exact (fun H => H).
Qed.

(* What a primitive does: what [eval_prim] gives it, the heap as it
   came; or one of the heap; or a read of the cell of a location that is
   its first argument; or a comparison with a location, a constant. *)
Lemma exec_prim_cases (statics : static_env) (h h' : heap) (p : prim)
    (vs : list value) (v : value) :
  exec_prim statics h p vs = Some (v, h') ->
  (eval_prim statics p vs = Some v ∧ h' = h)
  ∨ (∃ q : heap_primitive, p = Heap q)
  ∨ (∃ (up : unary_primitive) (l : location) (c : cell),
       p = Unary up ∧ vs = [Value_location l] ∧ lookup l h = Some c
       ∧ exec_unary_on_cell up l c = Some v ∧ h' = h)
  ∨ (∃ (op : equality_comparison) (c : const),
       p = Binary (Phys_equal op) ∧ v = Value_const c ∧ h' = h)
  ∨ (∃ (b : binary_primitive) (l : location) (c : cell) (v2 : value),
       p = Binary b ∧ vs = [Value_location l; v2] ∧ lookup l h = Some c
       ∧ exec_binary_on_cell b l c v2 = Some v ∧ h' = h).
Proof.
  unfold exec_prim.
  destruct (forallb (argument_bound statics h) vs); [ | discriminate ].
  unfold exec_bound_prim.
  Local Ltac eval_case :=
    let H := fresh "H" in
    intros H; cbv beta iota zeta in H;
    repeat match type of H with
           | context [match ?w with _ => _ end] =>
             is_var w; destruct w; cbv beta iota zeta in H
           end;
    revert H;
    match goal with
    | |- context [eval_prim ?st ?p ?vs] =>
      let E := fresh "E" in
      destruct (eval_prim st p vs) eqn:E; intros H; cbv beta iota zeta in H;
      [ injection H as <- <-; left; split;
        [ first [ exact E | reflexivity ] | reflexivity ]
      | discriminate H ]
    end.
  destruct p as [up | b | vp | q | r];
    [ | | eval_case | intros _; right; left; eexists; reflexivity
    | eval_case ].
  - destruct vs as [ | v1 [ | v2 vs]]; [ eval_case | | eval_case ].
    destruct v1; try eval_case.
    intros H. destruct (lookup l h) as [c | ] eqn:Hl; simpl in H;
      [ | discriminate H ].
    destruct (exec_unary_on_cell up l c) as [w | ] eqn:Hw; simpl in H;
      [ | discriminate H ].
    injection H as <- <-. right; right; left. exists up, l, c. auto.
  - destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; [ eval_case | eval_case | |
      destruct b; eval_case ].
    destruct b as [ak lk lm | w | op | ik op | ik op | ik bh ];
      destruct v1; try eval_case;
      intros H; cbv beta iota zeta in H; revert H.
    (* What is left: [Phys_equal], and a location as first argument. *)
    all: first
      [ match goal with
        | |- context [if ?c then _ else _] =>
          destruct c;
          [ match goal with
            | |- context [eval_phys_equal_location ?st ?x ?y] =>
              destruct (eval_phys_equal_location st x y); intros H;
              cbv beta iota zeta in H;
              [ injection H as <- <-; right; right; right; left;
                do 2 eexists; split; [ reflexivity | split; reflexivity ]
              | discriminate H ]
            end
          | eval_case ]
        end
      | match goal with
        | |- context [lookup ?l ?h] =>
          destruct (lookup l h) as [c | ] eqn:Hl; intros H;
          cbv beta iota zeta in H; [ revert H | discriminate H ];
          match goal with
          | |- context [exec_binary_on_cell ?b l c ?v2] =>
            destruct (exec_binary_on_cell b l c v2) eqn:Hw; intros H;
            cbv beta iota zeta in H;
            [ injection H as <- <-; right; right; right; right;
              exists b, l, c, v2; auto
            | discriminate H ]
          end
        end ].
Qed.

(* The cell of a location that a primitive reads, through the variable
   [y] that holds it. *)
Lemma read_location (vars : env) (args : list simple) (l : location)
    (rest : list value) :
  env_flows_ vars ->
  eval_simples vars args = Some (Value_location l :: rest) ->
  ∃ (y : variable) (args' : list simple) (a : node),
    args = Var y :: args' ∧ located l = Some a
    ∧ may_come_from sol a (Node_variable y)
    ∧ flows_ (Value_location l) (Node_variable y).
Proof.
  intros Hvars Hvs.
  destruct args as [ | s1 args']; simpl in Hvs; [ discriminate Hvs | ].
  destruct (eval_simple vars s1) as [w | ] eqn:Hs1; [ | discriminate Hvs ].
  destruct (eval_simples vars args') as [ws | ]; [ | discriminate Hvs ].
  injection Hvs as -> _.
  destruct (location_simple _ _ _ Hs1) as (y & ->).
  pose proof (Hvars y _ Hs1) as Hy.
  destruct (flows_location _ _ Hy) as (a & Ha & Hay).
  exists y, args', a. auto.
Qed.

(* The value of a [Let] of a primitive is at its variable, and the heap
   after holds what it says. *)
Lemma prim_exec_flows (statics : static_env) (vars : env) (hp hp' : heap)
    (ctx : option code_id) (x : variable) (p : prim) (args : list simple)
    (vs : list value) (v : value) :
  statics_flow_ statics -> env_flows_ vars -> heap_flows_ hp -> live ctx ->
  named_edges G noted deps keep slots heap_prims ctx x (Prim p args) ->
  (∀ q : heap_primitive,
     p = Heap q -> heap_allocates q = true ->
     located (fresh_location hp) = Some (Node_variable x)) ->
  eval_simples vars args = Some vs ->
  exec_prim statics hp p vs = Some (v, hp') ->
  flows_ v (Node_variable x) ∧ heap_flows_ hp'.
Proof.
  intros Hst Hvars Hheap Hlive Hnamed Halloc Hvs Hexec.
  pose proof Hnamed as Hedges.
  simpl in Hedges. unfold prim_edges in Hedges.
  destruct Hedges as [(_ & Heff & Hmut) Hedges].
  (* The symbols among the arguments are bound. *)
  assert (Hbound : ∀ (s : simple) (sym : symbol),
            s ∈ args -> s = Symbol sym ->
            lookup sym (static_values statics) <> None).
  { intros s sym Hin ->.
    apply (FlambdaHeapProofs.exec_prim_symbols_bound _ _ _ _ _ _ _ Hexec).
    destruct (In_nth_error _ _ Hin) as (i & Hi).
    destruct (nth_error vs i) as [w | ] eqn:Hw.
    - destruct (eval_simples_nth _ _ _ _ _ Hvs Hw) as (s & Hs & Hsw).
      rewrite Hi in Hs. injection Hs as <-. simpl in Hsw.
      injection Hsw as <-. exact (nth_error_In _ _ Hw).
    - apply nth_error_None in Hw.
      rewrite (eval_simples_length _ _ _ Hvs) in Hw.
      assert (Hlt : i < length args)
        by (apply nth_error_Some; rewrite Hi; discriminate).
      lia. }
  destruct (exec_prim_cases _ _ _ _ _ _ Hexec)
    as [[Hev ->] | [(q & ->) | [(up & l & c & -> & -> & Hl & Hcell & ->)
       | [(op & k & -> & -> & ->)
         | (b & l & c & v2 & -> & -> & Hl & Hcell & ->)]]]].
  - (* As [eval_prim] has it. *)
    split; [ | exact Hheap ].
    apply (named_flows statics vars ctx x (Prim p args) v Hst Hvars Hnamed).
    simpl. rewrite Hvs. exact Hev.
  - (* A primitive of the heap. *)
    unfold exec_prim, exec_bound_prim in Hexec.
    destruct (forallb (argument_bound statics hp) vs); [ | discriminate ].
    destruct q as [mut tag | kind init fld | kind | kind set_kind];
      cbn [exec_heap_primitive] in Hexec.
    + (* An allocation of a block. *)
      destruct (forallb (fun w => kind_eqb (kind_of_value w) Value) vs);
        [ | discriminate Hexec ].
      injection Hexec as <- <-.
      pose proof (Halloc _ eq_refl eq_refl) as Hx.
      simpl in Hedges. destruct Hedges as (Hcons & His & Hgt & Hnoted).
      split.
      * simpl. rewrite Hx.
        exact (alloc_self _ _ _ His (as_hs _ _ _ Hc _ consts_any_source)).
      * intros l' c' a Hl' Ha. rewrite lookup_cons in Hl'.
        destruct (Pos.eqb_spec l' (fresh_location hp)) as [-> | _];
          [ | exact (Hheap l' c' a Hl' Ha) ].
        injection Hl' as <-. rewrite Hx in Ha. injection Ha as <-.
        destruct (block_structure_made _ _ _ _
                    (conj Hcons (conj His (conj Hgt Hnoted))) Hvars Hvs)
          as (_ & _ & Hfields).
        split; [ exact His | split; [ exact Hgt | ] ].
        revert Hfields. apply fields_from_impl.
        intros i field _ (from & Hcf & Hf). exists from.
        split; [ exact Hcf | left; exact Hf ].
    + (* A store into a block. *)
      destruct vs as [ | v1 [ | new [ | v3 vs]]]; try discriminate Hexec.
      destruct v1; try discriminate Hexec.
      destruct (lookup l hp) as [[[ | ] tag fields | ak elements] | ] eqn:Hl;
        try discriminate Hexec.
      destruct (store_field kind fld new tag fields) as [fields' | ] eqn:Hsf;
        simpl in Hexec; [ | discriminate Hexec ].
      injection Hexec as <- <-.
      simpl in Hedges. destruct Hedges as [Has Huse].
      pose proof (cond_use_live ctx _ Hlive (Heff eq_refl)) as Hau.
      destruct (read_location _ _ _ _ Hvars Hvs)
        as (y & args' & a & -> & Ha & Hay & _).
      assert (Haua : au a).
      { apply (proj1 (proj2 (proj2 (proj2 Hay)))).
        exact (any_usage_use _ _ _ Hc _ _ (au_hu _ _ _ Hc _ Hau)
                 (Huse _ (or_introl eq_refl))). }
      destruct (eval_simples_nth _ _ _ 1 new Hvs eq_refl) as (s2 & Hs2 & Hev2).
      assert (Hesc : escapes G sol keep called located new).
      { apply (simple_escapes statics vars x (Var y :: args') s2 new Hst Hvars
                 (nth_error_In _ _ Hs2) Huse); [ | exact Hau | exact Hev2 ].
        intros sym Hsym. exact (Hbound s2 sym (nth_error_In _ _ Hs2) Hsym). }
      split; [ exact (any_source_input _ _ _ Hc _ Has) | ].
      intros l' c' a' Hl' Ha'. rewrite lookup_cons in Hl'.
      destruct (Pos.eqb_spec l' l) as [-> | _];
        [ | exact (Hheap l' c' a' Hl' Ha') ].
      injection Hl' as <-. rewrite Ha in Ha'. injection Ha' as <-.
      destruct (Hheap l _ a Hl Ha) as (His & Hgt & Hfields).
      split; [ exact His | split; [ exact Hgt | ] ].
      apply (fields_from_replace _ 0 (Z.to_nat fld) new fields fields' Hfields
               (store_field_replace _ _ _ _ _ _ Hsf)).
      intros old _ (from & Hcons & _). exists from.
      split; [ exact Hcons | right ].
      split; [ reflexivity | split; [ exact Haua | exact Hesc ] ].
    + (* An allocation of an array. *)
      destruct vs as [ | v0 vs0]; [ discriminate Hexec | ].
      destruct (array_made kind (v0 :: vs0)) as [elements | ] eqn:Ham;
        simpl in Hexec; [ | discriminate Hexec ].
      injection Hexec as <- <-.
      destruct (FlambdaPrimitiveProofs.array_made_of_arguments _ _ _ Ham)
        as [-> _].
      pose proof (Halloc _ eq_refl eq_refl) as Hx.
      simpl in Hedges. destruct Hedges as [Has Huse].
      split.
      * simpl. rewrite Hx.
        exact (any_source_self _ (any_source_input _ _ _ Hc _ Has)).
      * intros l' c' a Hl' Ha. rewrite lookup_cons in Hl'.
        destruct (Pos.eqb_spec l' (fresh_location hp)) as [-> | _];
          [ | exact (Hheap l' c' a Hl' Ha) ].
        injection Hl' as <-. rewrite Hx in Ha. injection Ha as <-.
        unfold cell_flows. apply (every_in _ _ _). intros w Hw.
        destruct (eval_simples_in _ _ _ _ Hvs Hw) as (s & Hin & Hs).
        destruct (simple_used statics vars x args s w Hst Hvars Hin Huse
                    (fun sym Hsym => Hbound s sym Hin Hsym) Hs)
          as [Hconst | (m & Hm & Hmu)]; [ left; exact Hconst | ].
        right. exists m. split; [ exact Hm | ].
        intros Hau. exact (Hmu (au_hu _ _ _ Hc _ Hau)).
    + (* A store into an array. *)
      destruct vs as [ | v1 [ | idx [ | new [ | v4 vs]]]];
        try discriminate Hexec.
      destruct v1; try discriminate Hexec.
      destruct (lookup l hp) as [[mut tag fields | ak elements] | ] eqn:Hl;
        try discriminate Hexec.
      destruct (array_stored set_kind ak elements idx new) as [elements' | ]
        eqn:Has'; simpl in Hexec; [ | discriminate Hexec ].
      injection Hexec as <- <-.
      simpl in Hedges. destruct Hedges as [Has Huse].
      pose proof (cond_use_live ctx _ Hlive (Heff eq_refl)) as Hau.
      destruct (read_location _ _ _ _ Hvars Hvs)
        as (y & args' & a & -> & Ha & Hay & _).
      destruct (eval_simples_nth _ _ _ 2 new Hvs eq_refl) as (s3 & Hs3 & Hev3).
      destruct (simple_used statics vars x (Var y :: args') s3 new Hst Hvars
                  (nth_error_In _ _ Hs3) Huse
                  (fun sym Hsym => Hbound s3 sym (nth_error_In _ _ Hs3) Hsym)
                  Hev3)
        as [Hconst | (m & Hm & Hmu)].
      * split; [ exact (any_source_input _ _ _ Hc _ Has) | ].
        intros l' c' a' Hl' Ha'. rewrite lookup_cons in Hl'.
        destruct (Pos.eqb_spec l' l) as [-> | _];
          [ | exact (Hheap l' c' a' Hl' Ha') ].
        injection Hl' as <-. rewrite Ha in Ha'. injection Ha' as <-.
        pose proof (Hheap l _ a Hl Ha) as Hold.
        unfold cell_flows in Hold |- *.
        apply (every_in _ _ _). intros w Hw.
        destruct (FlambdaPrimitiveProofs.array_stored_elements _ _ _ _ _ _ Has')
          as (_ & _ & Hstored).
        destruct (Hstored w Hw) as [-> | Hin]; [ left; exact Hconst | ].
        exact (proj1 (every_in _ _ _) Hold w Hin).
      * split; [ exact (any_source_input _ _ _ Hc _ Has) | ].
        intros l' c' a' Hl' Ha'. rewrite lookup_cons in Hl'.
        destruct (Pos.eqb_spec l' l) as [-> | _];
          [ | exact (Hheap l' c' a' Hl' Ha') ].
        injection Hl' as <-. rewrite Ha in Ha'. injection Ha' as <-.
        pose proof (Hheap l _ a Hl Ha) as Hold.
        unfold cell_flows in Hold |- *.
        apply (every_in _ _ _). intros w Hw.
        destruct (FlambdaPrimitiveProofs.array_stored_elements _ _ _ _ _ _ Has')
          as (_ & _ & Hstored).
        destruct (Hstored w Hw) as [-> | Hin];
          [ | exact (proj1 (every_in _ _ _) Hold w Hin) ].
        right. exists m. split; [ exact Hm | ].
        intros _. exact (Hmu (au_hu _ _ _ Hc _ Hau)).
  - (* A unary primitive reads the cell of a location. *)
    split; [ | exact Hheap ].
    destruct (read_location _ _ _ _ Hvars Hvs)
      as (y & args' & a & -> & Ha & Hay & Hy).
    destruct args' as [ | s' args'];
      [ | pose proof (eval_simples_length _ _ _ Hvs) as Hlen;
          simpl in Hlen; discriminate Hlen ].
    pose proof (Hheap l c a Hl Ha) as Hcf.
    (* A constant at a variable of the default case. *)
    Local Ltac const_at_default Hedges :=
      simpl in Hedges; destruct Hedges as [Has _];
      exact (any_source_input _ _ _ Hc _ Has).
    destruct c as [mut tag fields | ak elements];
      cbn [exec_unary_on_cell] in Hcell.
    + destruct up as [kind lmut fld | [ | ] | | | ak | | ok | ik op | src dst
                    | | bn | bn | | | r];
        try discriminate Hcell.
      * (* [Block_load] *)
        destruct (load_allowed lmut mut) eqn:Hla; [ | discriminate Hcell ].
        pose proof (load_field_nth _ _ _ _ _ Hcell) as Hnth.
        destruct Hcf as (_ & _ & Hfields).
        destruct (fields_from_elim _ _ _ _ _ Hfields Hnth)
          as (from & Hcons & [Hfrom | (Hm & _ & Hesc)]).
        -- simpl in Hedges. destruct Hedges as [Hacc _].
           refine (flows_read _ _ _ _ _ _ Hc _ _ _ _ _ _ _ Hcons Hacc Hay
                     Hfrom).
           not_local_field.
        -- subst mut. destruct lmut; try discriminate Hla.
           apply (escapes_flows v _ Hesc).
           exact (any_source_input _ _ _ Hc _ (Hmut eq_refl)).
      * (* [Is_int] of a variant *)
        injection Hcell as <-. destruct Hcf as (His & _ & _).
        simpl in Hedges. destruct Hedges as [Hacc _].
        refine (read_const a _ x Field_is_int (Naked_immediate 0) _ His Hacc
                  Hay).
        not_local_field.
      * injection Hcell as <-. const_at_default Hedges.
      * injection Hcell as <-. const_at_default Hedges.
      * (* [Get_tag] *)
        injection Hcell as <-. destruct Hcf as (_ & Hgt & _).
        simpl in Hedges. destruct Hedges as [Hacc _].
        refine (read_const a _ x Field_get_tag (Naked_immediate tag) _ Hgt
                  Hacc Hay).
        not_local_field.
      * (* [Opaque_identity] *)
        destruct (kind_eqb ok Value); [ | discriminate Hcell ].
        injection Hcell as <-.
        simpl in Hedges. destruct Hedges as [Has Huse].
        apply (flows_escape _ _ _ _ _ _ Hc _ _ (Node_variable y)
                 (any_source_input _ _ _ Hc _ Has)); [ | exact Hy ].
        intros Hhu.
        exact (any_usage_use _ _ _ Hc _ _ Hhu (Huse _ (or_introl eq_refl))).
    + destruct up as [kind lmut fld | [ | ] | | | ak' | | ok | ik op | src dst
                    | | bn | bn | | | r];
        try discriminate Hcell.
      * injection Hcell as <-. const_at_default Hedges.
      * injection Hcell as <-. const_at_default Hedges.
      * unfold array_length_of in Hcell.
        destruct (in_range _ _); [ | discriminate Hcell ].
        injection Hcell as <-. const_at_default Hedges.
      * (* [Opaque_identity] *)
        destruct (kind_eqb ok Value); [ | discriminate Hcell ].
        injection Hcell as <-.
        simpl in Hedges. destruct Hedges as [Has Huse].
        apply (flows_escape _ _ _ _ _ _ Hc _ _ (Node_variable y)
                 (any_source_input _ _ _ Hc _ Has)); [ | exact Hy ].
        intros Hhu.
        exact (any_usage_use _ _ _ Hc _ _ Hhu (Huse _ (or_introl eq_refl))).
  - (* A comparison with a location. *)
    split; [ | exact Hheap ]. const_at_default Hedges.
  - (* A load from an array. *)
    split; [ | exact Hheap ].
    destruct (read_location _ _ _ _ Hvars Hvs)
      as (y & args' & a & -> & Ha & Hay & _).
    pose proof (Hheap l c a Hl Ha) as Hcf.
    destruct c as [mut tag fields | ak elements];
      [ destruct b; discriminate Hcell | ].
    destruct b as [ak' lk lm | w | op | ik op | ik op | ik bh ];
      cbn [exec_binary_on_cell] in Hcell; try discriminate Hcell.
    destruct (load_allowed lm Cell_mutable); [ | discriminate Hcell ].
    destruct (FlambdaPrimitiveProofs.array_loaded_element _ _ _ _ Hcell)
      as [Hin _].
    simpl in Hedges. destruct Hedges as [Has Huse].
    destruct (proj1 (every_in _ _ _) Hcf v Hin)
      as [(k & ->) | (m & Hm & Hmu)];
      [ exact (any_source_input _ _ _ Hc _ Has) | ].
    apply (flows_escape _ _ _ _ _ _ Hc v _ m
             (any_source_input _ _ _ Hc _ Has)); [ | exact Hm ].
    intros Hhu. apply Hmu. apply (proj1 (proj2 (proj2 (proj2 Hay)))).
    exact (any_usage_use _ _ _ Hc _ _ Hhu (Huse _ (or_introl eq_refl))).
Qed.

(* What the cells hold is as the run makes it after a primitive. *)
Lemma prim_exec_pa_ok (statics : static_env) (vars : env) (hp hp' : heap)
    (p : prim) (args : list simple) (vs : list value) (v : value) :
  env_pa_ok statics vars -> statics_pa_ok statics ->
  heap_pa_ok statics hp ->
  eval_simples vars args = Some vs ->
  exec_prim statics hp p vs = Some (v, hp') ->
  pa_ok statics v ∧ heap_pa_ok statics hp'.
Proof.
  intros Hvars Hst Hheap Hvs Hexec.
  pose proof (simples_pa_ok _ _ _ _ Hvars Hvs) as Hvs_pa.
  assert (Hcase : (∃ q : heap_primitive, p = Heap q)
                  ∨ ∀ q : heap_primitive, p <> Heap q).
  { destruct p as [up | b | vp | q | r];
      [ right | right | right | left; exists q; reflexivity | right ];
      intros q' H; discriminate H. }
  destruct Hcase as [(q & ->) | Hnot].
  - destruct (FlambdaHeapProofs.exec_prim_adds _ _ _ _ _ _ Hexec)
      as [(c & Hc_vals & -> & ->)
         | (l & rest & new & c & c' & i & -> & Hl & Hcw & -> & ->)].
    + split; [ exact I | ].
      intros l' c'' w Hl' Hw. rewrite lookup_cons in Hl'.
      destruct (Pos.eqb l' (fresh_location hp));
        [ injection Hl' as <- | exact (Hheap _ _ _ Hl' Hw) ].
      rewrite Hc_vals in Hw. exact (proj1 (every_in _ _ _) Hvs_pa w Hw).
    + split; [ exact I | ].
      intros l' c'' w Hl' Hw. rewrite lookup_cons in Hl'.
      destruct (Pos.eqb l' l);
        [ injection Hl' as <- | exact (Hheap _ _ _ Hl' Hw) ].
      destruct (FlambdaHeapProofs.cell_with_values _ _ _ _ _ Hcw Hw)
        as [-> | Hin]; [ | exact (Hheap _ _ _ Hl Hin) ].
      apply (proj1 (every_in _ _ _) Hvs_pa). right. apply in_or_app.
      right. left. reflexivity.
  - pose proof (FlambdaHeapProofs.exec_prim_heap _ _ _ _ _ _ Hnot Hexec)
      as Heq.
    subst hp'. split; [ | exact Hheap ].
    refine (FlambdaHeapProofs.exec_prim_preserves_given (pa_ok statics)
              statics hp hp p vs v _ _ _ _ _ Hnot _ _ Hexec).
    + intros c. split; exact I.
    + intros tag fields. simpl. apply (every_in _ _ _).
    + intros elements. simpl. apply (every_in _ _ _).
    + intros s tag fields _ Hs w Hw. specialize (Hst s _ Hs). simpl in Hst.
      exact (proj1 (every_in _ _ _) Hst w Hw).
    + intros s elements _ Hs w Hw. specialize (Hst s _ Hs). simpl in Hst.
      exact (proj1 (every_in _ _ _) Hst w Hw).
    + exact (proj1 (every_in _ _ _) Hvs_pa).
    + intros l c w _ Hl Hw. exact (Hheap l c w Hl Hw).
Qed.

(* A step keeps the invariant, and an exit has its values at the nodes
   of the continuation of the unit: at a [Let] if the state holds no
   location and the term no primitive of the heap. *)
Lemma step_flow_invariant_core (s : state) :
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
     state_expr s <> Let x kind d body)
  ∨ (~ heap_prims ∧ ∀ l : location, located l = None) ->
  flow_invariant s ->
  match step s with
  | Step s' => flow_invariant s'
  | Exit k vs _ _ => exit_flows G sol keep called located u k vs
  | Undefined => True
  end.
Proof.
  intros Hcase (ctx & ce & He & Hlive & Hcount & Hvars & Hconts & Hst & Hcodes
          & Hvars_pa & Hspa & HK_pa).
  rewrite (state_eta s).
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl in He; unfold counted in Hcount; simpl in Hcount.
  - (* [Let] *)
    destruct Hcase as [Hnot | [Hnh Hnl]];
      [ destruct (Hnot x kind d body eq_refl) | ].
    destruct He as [Hd Hbody].
    rewrite step_let_eval;
      [ | exact (named_edges_no_heap _ _ _ _ Hnh Hd)
        | intros p args vs l _ Hvs;
          exact (env_flows_no_location _ _ _ l (Hnl l) Hvars Hvs) ].
    destruct (eval_named (state_statics s) (state_vars s) d) as [v | ] eqn:Hev;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    exists ctx, ce. simpl_state.
    split; [ exact Hbody | split; [ exact Hlive | split; [ exact Hcount | ] ] ].
    split; [ | split; [ exact Hconts | split; [ exact Hst | split;
      [ exact Hcodes | ] ] ] ].
    + intros y w Hy. simpl in Hy. destruct (Pos.eqb_spec y x) as [-> | _].
      * injection Hy as <-.
        exact (named_flows _ _ _ _ _ _ Hst Hvars Hd Hev).
      * exact (Hvars y w Hy).
    + split; [ | split; [ exact Hspa | exact HK_pa ] ].
      intros y w Hy. simpl in Hy. destruct (Pos.eqb_spec y x) as [-> | _].
      * injection Hy as <-. exact (named_pa_ok _ _ _ _ Hvars_pa Hspa Hev).
      * exact (Hvars_pa y w Hy).
  - (* [Let_static] *)
    rewrite step_let_static.
    destruct (eval_static_bindings (state_vars s) bindings (state_statics s))
      as [statics' | ] eqn:Hb; [ | exact I ].
    destruct He as [Hbindings Hbody].
    destruct (statics_bindings _ _ _ _ Hvars Hbindings Hst Hcodes Hb)
      as [Hst' Hcodes'].
    pose proof (eval_static_bindings_le _ _ _ _ Hb) as Hle.
    exists ctx, ce. simpl_state.
    split; [ exact Hbody | split; [ exact Hlive | split; [ exact Hcount | ] ] ].
    split; [ exact Hvars | split; [ exact Hconts | split; [ exact Hst'
      | split; [ exact Hcodes' | ] ] ] ].
    split; [ exact (env_pa_ok_le _ _ _ Hle Hvars_pa) | split ].
    + exact (statics_bindings_pa _ _ _ _ Hbindings Hvars_pa Hspa Hb).
    + intros k' c Hin. exact (cont_pa_ok_le _ _ _ Hle (HK_pa k' c Hin)).
  - (* [Let_cont_non_recursive] *)
    rewrite step_let_cont_non_recursive.
    destruct He as [Hh Hbody]. exists ctx, ((k, handler_params h) :: ce).
    simpl_state.
    split; [ exact Hbody | split; [ exact Hlive | split ] ].
    { intros Hctx. specialize (Hcount Hctx).
      apply andb_true_iff in Hcount as [_ Hcount]. exact Hcount. }
    split; [ exact Hvars | split; [ | split; [ exact Hst | split;
      [ exact Hcodes | ] ] ] ].
    + apply Conts_cons; [ | exact Hconts ].
      apply (Closure_ok h _ _ [] _ ctx ce Hvars Hconts Hlive).
      * intros k' h' [].
      * exact Hh.
      * intros Hctx. specialize (Hcount Hctx).
        apply andb_true_iff in Hcount as [Hcount _]. exact Hcount.
      * intros i v n p Hn Hp. rewrite Hn in Hp. injection Hp as <-. auto.
    + split; [ exact Hvars_pa | split; [ exact Hspa | ] ].
      intros k' c [[= <- <-] | Hin]; [ | exact (HK_pa k' c Hin) ].
      apply Closure_pa; [ exact Hvars_pa | exact HK_pa ].
  - (* [Let_cont_recursive] *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    destruct He as [Hhandlers Hbody]. exists ctx, (rec_conts ce handlers).
    simpl_state.
    split; [ exact Hbody | split; [ exact Hlive | split ] ].
    { intros Hctx. specialize (Hcount Hctx).
      apply andb_true_iff in Hcount as [_ Hcount].
      rewrite counts_rec. exact Hcount. }
    split; [ exact Hvars | split; [ | split; [ exact Hst | split;
      [ exact Hcodes | ] ] ] ].
    + apply conts_ok_rec; [ exact Hvars | exact Hconts | exact Hlive | ].
      intros k' h' Hin. split.
      * exact (proj1 (every_in _ _ _) Hhandlers (k', h') Hin).
      * intros Hctx. specialize (Hcount Hctx).
        apply andb_true_iff in Hcount as [Hcount _].
        rewrite forallb_forall in Hcount.
        unfold handler_counted. rewrite counts_rec.
        exact (Hcount (k', h') Hin).
    + split; [ exact Hvars_pa | split; [ exact Hspa | ] ].
      intros k' c Hin. apply in_app_or in Hin as [Hin | Hin];
        [ | exact (HK_pa k' c Hin) ].
      unfold rec_closures in Hin.
      apply in_map_iff in Hin as ([k'' h''] & [= <- <-] & _).
      apply Closure_pa; [ exact Hvars_pa | exact HK_pa ].
  - (* [Apply] *)
    destruct He as (Hfrag & Hnoted_callee & Hnoted_args & returns & exn & Hret
                    & Hexn & Hcall).
    destruct (ap_call_kind ap) as [id | | code_ids] eqn:Hkind;
      [ | | (* Not in the fragment. *)
            unfold apply_in_flow_fragment in Hfrag; rewrite Hkind in Hfrag;
            discriminate Hfrag ].
    + (* A direct call. *)
      pose proof (call_step ap (state_vars s) (state_conts s) (state_statics s)
                    (state_heap s) (state_reads s) id Hkind Hfrag Hcodes) as
                      Hstep.
      destruct (step _) as [s' | | ]; [ | contradiction | exact I ].
      destruct Hstep as (callee & vs & extra & slots & c & callee_vars
                         & Hcallee & Hvs & Hextra & Hclosure & Hcode & Hkinds
                         & Hparams & Hresult & Hbind & ->).
      destruct (apply_conts ctx ce ap (state_vars s) (state_conts s)
                  (state_statics s) extra returns exn Hcount Hvars Hvars_pa
                  Hconts HK_pa Hret Hexn Hextra)
        as ((Hret_ok & Hret_fits & Hret_pa) & Hexn_ok & Hexn_pa).
      exact (direct_call_ok ctx ap (state_vars s) (state_statics s)
               (state_heap s) (state_reads s) id callee vs slots c callee_vars
                 returns exn _ _
               Hkind Hcall Hnoted_callee Hnoted_args Hlive Hvars Hvars_pa Hst
               Hcodes Hspa Hret_ok Hret_fits Hret_pa Hexn_ok Hexn_pa Hcallee
               Hvs Hclosure Hcode Hkinds Hparams Hresult Hbind).
    + (* An indirect call. *)
      rewrite step_apply.
      destruct (eval_simple (state_vars s) (ap_callee ap)) as [callee | ]
        eqn:Hcallee; [ | exact I ].
      destruct (eval_simples (state_vars s) (ap_args ap)) as [vs | ] eqn:Hvs;
        [ | exact I ].
      destruct (eval_simples (state_vars s) (exn_extra_simples ap))
        as [extra | ] eqn:Hextra; [ | exact I ].
      destruct (apply_conts ctx ce ap (state_vars s) (state_conts s)
                  (state_statics s) extra returns exn Hcount Hvars Hvars_pa
                  Hconts HK_pa Hret Hexn Hextra)
        as ((Hret_ok & Hret_fits & Hret_pa) & Hexn_ok & Hexn_pa).
      rewrite Hkind.
      unfold call_edges in Hcall. rewrite Hkind in Hcall.
      destruct Hcall as (ws & apply_node & called_node & Hlen & Hch & Hused
                         & Hcond & Happly).
      assert (Hind : indirect_ok).
      { intros id c Hin. unfold apply_in_flow_fragment in Hfrag.
        rewrite Hkind in Hfrag. rewrite forallb_forall in Hfrag.
        specialize (Hfrag (id, c) Hin). simpl in Hfrag.
        destruct (code_is_tupled c); [ exact Hfrag | ].
        apply negb_true_iff in Hfrag.
        intros H; rewrite H in Hfrag; discriminate Hfrag. }
      change (result_ok
                (call_function callee Indirect_unknown_arity vs
                   (ap_args_arity ap) (ap_return_arity ap)
                   (resolve (ap_continuation ap) (state_conts s))
                   (with_extra_args
                      (resolve (ap_exn_continuation ap) (state_conts s))
                      extra)
                   (state_statics s) (state_heap s) (state_reads s))).
      apply (call_ok callee (node_of_simple (ap_callee ap)) vs
               (ap_args_arity ap) (ap_return_arity ap) ws returns exn _ _
               (state_statics s) (state_heap s) (state_reads s) Hind); try
                 assumption.
      * exact (simple_flows _ _ _ Hvars Hnoted_callee Hcallee).
      * exact (simple_pa_ok _ _ _ _ Hvars_pa Hcallee).
      * intros Hne.
        exact (indirect_pending ctx ap (state_vars s) vs returns exn ws
                 apply_node called_node Hlive Hvars Hnoted_args Hvs Hne Hlen
                 Hch Hused Hcond Happly).
      * exact (simples_pa_ok _ _ _ _ Hvars_pa Hvs).
      * intros v Hv Hv_pa Hone.
        apply (apply_ok _ returns [v] _ _ _ Hret_ok Hret_pa); try assumption.
        -- intros [ | i] x r Hx Hr; [ injection Hx as <-; exact (Hv r Hr)
                                    | destruct i; discriminate Hx ].
        -- split; [ exact Hv_pa | exact I ].
        -- destruct Hret_fits as [Hf | Hf]; [ left | right; exact Hf ].
           rewrite <- Hf, Hone. reflexivity.
  - (* [Apply_cont] *)
    rewrite step_apply_cont.
    destruct (eval_simples (state_vars s) (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    pose proof (eval_simples_length _ _ _ Hvs) as Hlen_vs.
    unfold apply_continuation, resolve.
    destruct (lookup (ac_continuation ac) (state_conts s)) as [c | ] eqn:Hl.
    + destruct (conts_ok_lookup _ _ _ _ _ Hconts Hl) as (ns & Hns & Hok & Htop).
      pose proof (apply_cont_fits ctx ce ac ns c Hcount Hns Htop) as Hfits.
      pose proof (apply_cont_args_flow _ _ _ _ _ Hvars He Hvs Hns) as Hargs.
      change (result_ok (apply_cont_value c vs (state_statics s)
                           (state_heap s) (state_reads s))).
      apply (apply_ok c ns vs _ _ _ Hok (HK_pa _ c (lookup_in _ _ _ Hl)) Hargs);
        try assumption.
      * exact (simples_pa_ok _ _ _ _ Hvars_pa Hvs).
      * rewrite Hlen_vs. exact Hfits.
    + (* A way out of the unit, at toplevel. *)
      pose proof (conts_ok_none _ _ _ _ Hconts Hl) as Hnone.
      destruct ctx as [id | ].
      * exfalso. specialize (Hcount ltac:(discriminate)).
        destruct (count_ok_lookup _ _ _ Hcount) as (ns & Hns & _). congruence.
      * intros i v p Hv Hp. unfold unit_cont_nodes in Hp.
        destruct (lookup (ac_continuation ac) (unit_conts u)) as [l | ]
          eqn:Hu; [ | destruct i; discriminate Hp ].
        exact (apply_cont_args_flow _ _ _ _ _ Hvars He Hvs Hnone i v p Hv Hp).
  - (* [Switch] *)
    rewrite step_switch.
    destruct (eval_simple (state_vars s) scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    exists ctx, ce. simpl_state.
    split; [ exact (proj1 (every_in _ _ _) He (i, ac)
                      (find_arm_compact _ _ _ Harm)) | ].
    split; [ exact Hlive | split ].
    { intros Hctx. specialize (Hcount Hctx). rewrite forallb_forall in Hcount.
      exact (Hcount (i, ac) (find_arm_in _ _ _ Harm)). }
    split; [ exact Hvars | split; [ exact Hconts | split; [ exact Hst
      | split; [ exact Hcodes | ] ] ] ].
    split; [ exact Hvars_pa | split; [ exact Hspa | exact HK_pa ] ].
  - (* [Invalid] *)
    rewrite step_invalid. exact I.
Qed.

(* The step of a term without the primitives of the heap, in a state
   that holds no location. *)
Lemma step_flow_invariant (s : state) :
  ~ heap_prims -> (∀ l : location, located l = None) ->
  flow_invariant s ->
  match step s with
  | Step s' => flow_invariant s'
  | Exit k vs _ _ => exit_flows G sol keep called located u k vs
  | Undefined => True
  end.
Proof.
  intros Hnh Hnl.
  exact (step_flow_invariant_core s (or_intror (conj Hnh Hnl))).
Qed.

(* The static environment of a state only grows. *)
Lemma statics_le_refl (statics : static_env) : statics_le statics statics.
Proof. split; auto. Qed.

Lemma call_code_statics (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s' ->
  state_statics s' = statics.
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ discriminate | ].
  destruct (negb _); [ discriminate | ].
  destruct (bind_params _ _ _); [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

Lemma run_call_statics (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record) (s' : state) :
  (∀ (partial : value) (s' : state),
    return_partial partial = Step s' -> state_statics s' = statics) ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  state_statics s' = statics.
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply call_code_statics.
  - apply call_code_statics.
  - apply Hpartial.
  - discriminate.
Qed.

Lemma apply_cont_value_statics (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  apply_cont_value c vs statics hp rd = Step s' -> state_statics s' = statics.
Proof.
  revert vs s'.
  induction c
    as [handler vars conts recursive | k | c IH extra
       | remaining remaining_arity return_arity return_to IHreturn exn_to
           IHexn];
    intros vs s'; simpl.
  - destruct handler as [params metadata handler].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs vars); [ | discriminate ].
    intros [= <-]. reflexivity.
  - discriminate.
  - apply IH.
  - destruct vs as [ | v [ | v2 vs]]; try discriminate.
    apply run_call_statics. intros partial s0. apply IHreturn.
Qed.

Lemma call_function_statics (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record) (s' : state) :
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Step s' ->
  state_statics s' = statics.
Proof.
  unfold call_function. apply run_call_statics. intros partial s0.
  apply apply_cont_value_statics.
Qed.

Lemma step_statics_le (s s' : state) :
  step s = Step s' -> statics_le (state_statics s) (state_statics s').
Proof.
  rewrite (state_eta s).
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ]; simpl_state.
  - rewrite step_let_exec.
    destruct (exec_named _ _ _ _ d) as [[[v hp'] rd'] | ];
      [ | discriminate ].
    destruct (kind_eqb _ _); [ | discriminate ].
    intros [= <-]. apply statics_le_refl.
  - rewrite step_let_static.
    destruct (eval_static_bindings _ _ _) as [statics' | ] eqn:Hb;
      [ | discriminate ].
    intros [= <-]. exact (eval_static_bindings_le _ _ _ _ Hb).
  - rewrite step_let_cont_non_recursive. intros [= <-].
    apply statics_le_refl.
  - rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-]. apply statics_le_refl.
  - rewrite step_apply.
    destruct (eval_simple _ (ap_callee ap)); [ | discriminate ].
    destruct (eval_simples _ (ap_args ap)); [ | discriminate ].
    destruct (eval_simples _ (exn_extra_simples ap)); [ | discriminate ].
    intros Hstep. rewrite (call_function_statics _ _ _ _ _ _ _ _ _ _ _ Hstep).
    apply statics_le_refl.
  - rewrite step_apply_cont.
    destruct (eval_simples _ (ac_args ac)) as [vs | ]; [ | discriminate ].
    intros Hstep. unfold apply_continuation in Hstep.
    rewrite (apply_cont_value_statics _ _ _ _ _ _ Hstep).
    apply statics_le_refl.
  - rewrite step_switch.
    destruct (eval_simple _ scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. apply statics_le_refl.
  - rewrite step_invalid. discriminate.
Qed.

Lemma heap_pa_ok_le (statics statics' : static_env) (hp : heap) :
  statics_le statics statics' -> heap_pa_ok statics hp ->
  heap_pa_ok statics' hp.
Proof.
  intros Hle H l c v Hl Hv. exact (pa_ok_le _ _ _ Hle (H l c v Hl Hv)).
Qed.

(* What a step comes to keeps the invariant and that of the heap, or
   exits as the unit says with a heap that holds what it says. *)
Definition result_heap_ok (r : step_result) : Prop :=
  match r with
  | Step s' => flow_invariant s' ∧ heap_ok s'
  | Exit k vs hp _ => exit_flows G sol keep called located u k vs
                      ∧ heap_flows_ hp
  | Undefined => True
  end.

(* A step of a [Let]: the value is at the variable, and the heap after
   holds what it says. *)
Lemma let_flow_invariant (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  flow_invariant (State (Let x kind d body) vars conts statics hp rd) ->
  heap_ok (State (Let x kind d body) vars conts statics hp rd) ->
  alloc_ok (State (Let x kind d body) vars conts statics hp rd) ->
  result_heap_ok (step (State (Let x kind d body) vars conts statics hp rd)).
Proof.
  intros (ctx & ce & He & Hlive & Hcount & Hvars & Hconts & Hst & Hcodes
          & Hvars_pa & Hspa & HK_pa) [Hheap Hhpa] Halloc.
  simpl_state. simpl in He. unfold counted in Hcount. simpl in Hcount.
  destruct He as [Hd Hbody].
  rewrite step_let_exec.
  destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hexec;
    [ | exact I ].
  destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
  assert (Hv : flows_ v (Node_variable x) ∧ heap_flows_ hp'
               ∧ pa_ok statics v ∧ heap_pa_ok statics hp').
  { destruct d as [s0 | p args | | | | | ];
      try (rewrite exec_named_not_prim in Hexec;
           [ | intros p' args' Hp; discriminate Hp ];
           destruct (eval_named _ _ _) as [w | ] eqn:Hev; simpl in Hexec;
           [ injection Hexec as <- <- <- | discriminate Hexec ];
           split; [ exact (named_flows _ _ _ _ _ _ Hst Hvars Hd Hev) | ];
           split; [ exact Hheap | ];
           split; [ exact (named_pa_ok _ _ _ _ Hvars_pa Hspa Hev)
                  | exact Hhpa ]).
    apply exec_named_prim_some in Hexec as (vs & Hvs & Hprim & _).
    destruct (prim_exec_flows statics vars hp hp' ctx x p args vs v Hst Hvars
                Hheap Hlive Hd
                (fun q Hq Hal => Halloc x kind q args body
                                   ltac:(rewrite Hq; reflexivity) Hal)
                Hvs Hprim)
      as [Hf Hh].
    destruct (prim_exec_pa_ok statics vars hp hp' p args vs v Hvars_pa Hspa
                Hhpa Hvs Hprim)
      as [Hp Hhp].
    auto. }
  destruct Hv as (Hvx & Hheap' & Hpa & Hhpa').
  split; [ | split; [ exact Hheap' | exact Hhpa' ] ].
  exists ctx, ce. simpl_state.
  split; [ exact Hbody | split; [ exact Hlive | split; [ exact Hcount | ] ] ].
  split; [ | split; [ exact Hconts | split; [ exact Hst | split;
    [ exact Hcodes | ] ] ] ].
  - intros y w Hy. simpl in Hy. destruct (Pos.eqb_spec y x) as [-> | _].
    + injection Hy as <-. exact Hvx.
    + exact (Hvars y w Hy).
  - split; [ | split; [ exact Hspa | exact HK_pa ] ].
    intros y w Hy. simpl in Hy. destruct (Pos.eqb_spec y x) as [-> | _].
    + injection Hy as <-. exact Hpa.
    + exact (Hvars_pa y w Hy).
Qed.

(* A step keeps the invariant and that of the heap. *)
Lemma step_flow_invariant_heap (s : state) :
  flow_invariant s -> heap_ok s -> alloc_ok s -> result_heap_ok (step s).
Proof.
  intros Hinv Hh Halloc.
  destruct (state_expr s) as [x kind d body | | | | | | | ] eqn:Hexpr.
  1: { destruct s as [e vars conts statics hp rd]. simpl in Hexpr.
       subst e.
       exact (let_flow_invariant _ _ _ _ _ _ _ _ _ Hinv Hh Halloc). }
  all: assert (Hnot : ∀ (x : variable) (kind : flambda_kind) (d : named)
                        (body : expr), state_expr s <> Let x kind d body)
         by (intros x' kind' d' body' Hl; rewrite Hexpr in Hl;
             discriminate Hl).
  all: pose proof (step_flow_invariant_core s (or_introl Hnot) Hinv)
         as Hcore.
  all: destruct Hh as [Hheap Hhpa].
  all: unfold result_heap_ok;
       destruct (step s) as [s' | k0 vs0 hp0 rd0 | ] eqn:Hstep;
       [ | | exact I ].
  all: try (split; [ exact Hcore | ];
            rewrite (FlambdaSemanticsProofs.step_exit_heap _ _ _ _ _ Hstep);
            exact Hheap).
  all: split; [ exact Hcore | ].
  all: destruct (FlambdaSemanticsProofs.step_heap s s' Hstep)
         as [Heq | (x0 & kind0 & d0 & body0 & v0 & Hlet & _)];
       [ | rewrite Hexpr in Hlet; discriminate Hlet ].
  all: split; [ rewrite Heq; exact Hheap | ].
  all: rewrite Heq;
       exact (heap_pa_ok_le _ _ _ (step_statics_le _ _ Hstep) Hhpa).
Qed.

Lemma run_flow_invariant (n : nat) (s : state) :
  (∀ (i : nat) (s' : state), i < n -> run i s = Step s' -> alloc_ok s') ->
  flow_invariant s -> heap_ok s ->
  match run n s with
  | Step s' => flow_invariant s' ∧ heap_ok s'
  | Exit k vs hp _ =>
    exit_flows G sol keep called located u k vs ∧ heap_flows_ hp
  | Undefined => True
  end.
Proof.
  revert s. induction n as [ | n IH]; intros s Halloc Hs Hh;
    [ exact (conj Hs Hh) | ].
  rewrite run_succ.
  pose proof (step_flow_invariant_heap s Hs Hh
                (Halloc 0 s (Nat.lt_0_succ n) eq_refl)) as Hstep.
  destruct (step s) as [s' | k vs hp rd | ] eqn:E; [ | exact Hstep | exact I ].
  destruct Hstep as [Hs' Hh'].
  apply (IH s'); [ | exact Hs' | exact Hh' ].
  intros i s'' Hi Hrun. apply (Halloc (S i) s''); [ lia | ].
  rewrite run_succ, E. exact Hrun.
Qed.

Lemma flow_invariant_flows (s : state) :
  flow_invariant s ->
  env_flows_ (state_vars s)
  ∧ (∀ (k : continuation) (c : cont_value),
       (k, c) ∈ state_conts s -> cont_flows G sol keep called located c)
  ∧ statics_flow_ (state_statics s).
Proof.
  intros (ctx & ce & _ & _ & _ & Hvars & Hconts & Hst & _).
  split; [ exact Hvars | ].
  split; [ exact (proj1 conts_ok_flows _ _ _ Hconts) | exact Hst ].
Qed.

Lemma initial_flow_invariant (e : expr) :
  edges None (unit_conts u) e ->
  flow_invariant (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  intros He. exists None, (unit_conts u). simpl_state.
  split; [ exact He | split; [ exact I | split; [ intros H; contradiction | ]
    ] ].
  split; [ intros x v Hx; discriminate Hx | ].
  split; [ exact Conts_top | ].
  split; [ intros s allocated Hs; discriminate Hs | ].
  split; [ intros id c Hl; discriminate Hl | ].
  split; [ intros x v Hx; discriminate Hx | ].
  split; [ intros s allocated Hs; discriminate Hs | intros k c [] ].
Qed.

Lemma initial_heap_ok (e : expr) :
  heap_ok (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  split; [ intros l c a Hl; discriminate Hl
         | intros l c v Hl; discriminate Hl ].
Qed.

End Run.

End WithLocal.

(* The names of the run as before the fix: the same where the facts of a
   projection do not matter, and else where they carry that no field is
   local. *)
Abbreviation alias_consts_any_source := WithLocal.alias_consts_any_source.
Abbreviation alloc_ok := WithLocal.alloc_ok.
Abbreviation alloc_self := WithLocal.alloc_self.
Abbreviation any_source_self := WithLocal.any_source_self.
Abbreviation apply_cont_args_flow := WithLocal.apply_cont_args_flow.
Abbreviation apply_cont_fits := WithLocal.apply_cont_fits.
Abbreviation apply_cont_value_statics := WithLocal.apply_cont_value_statics.
Abbreviation args_at := WithLocal.args_at.
Abbreviation args_at_app := WithLocal.args_at_app.
Abbreviation args_at_skipn := WithLocal.args_at_skipn.
Abbreviation bind_params_length := WithLocal.bind_params_length.
Abbreviation block_fields_pa_ok := WithLocal.block_fields_pa_ok.
Abbreviation block_of_flows := WithLocal.block_of_flows.
Abbreviation block_structure := WithLocal.block_structure.
Abbreviation block_structure_made := WithLocal.block_structure_made.
Abbreviation boxed_flows := WithLocal.boxed_flows.
Abbreviation boxed_number_of_cases := WithLocal.boxed_number_of_cases.
Abbreviation boxed_self := WithLocal.boxed_self.
Abbreviation call_chain_app := WithLocal.call_chain_app.
Abbreviation call_chain_mono := WithLocal.call_chain_mono.
Abbreviation call_code_not_exit := WithLocal.call_code_not_exit.
Abbreviation call_code_statics := WithLocal.call_code_statics.
Abbreviation call_function_statics := WithLocal.call_function_statics.
Abbreviation called_alloc := WithLocal.called_alloc.
Abbreviation caller_call_chain := WithLocal.caller_call_chain.
Abbreviation caller_chain_nth := WithLocal.caller_chain_nth.
Abbreviation chain_cover := WithLocal.chain_cover.
Abbreviation chain_end_last := WithLocal.chain_end_last.
Abbreviation closure_alloc := WithLocal.closure_alloc.
Abbreviation code_chain_nth := WithLocal.code_chain_nth.
Abbreviation cond_alias_alias := WithLocal.cond_alias_alias.
Abbreviation cond_use_live := WithLocal.cond_use_live.
Abbreviation consts_any_source := WithLocal.consts_any_source.
Abbreviation cont_pa_closure_inv := WithLocal.cont_pa_closure_inv.
Abbreviation cont_pa_extra_inv := WithLocal.cont_pa_extra_inv.
Abbreviation cont_pa_over_inv := WithLocal.cont_pa_over_inv.
Abbreviation count_ok_lookup := WithLocal.count_ok_lookup.
Abbreviation counted := WithLocal.counted.
Abbreviation counts := WithLocal.counts.
Abbreviation counts_rec := WithLocal.counts_rec.
Abbreviation default_flows := WithLocal.default_flows.
Abbreviation env_flows_no_location := WithLocal.env_flows_no_location.
Abbreviation escapes_flows := WithLocal.escapes_flows.
Abbreviation every_app_intro := WithLocal.every_app_intro.
Abbreviation every_firstn := WithLocal.every_firstn.
Abbreviation every_skipn := WithLocal.every_skipn.
Abbreviation exec_prim_cases := WithLocal.exec_prim_cases.
Abbreviation fields_from_replace := WithLocal.fields_from_replace.
Abbreviation find_arm_compact := WithLocal.find_arm_compact.
Abbreviation firstn_length_app := WithLocal.firstn_length_app.
Abbreviation fits := WithLocal.fits.
Abbreviation flows_block := WithLocal.flows_block.
Abbreviation flows_location := WithLocal.flows_location.
Abbreviation handler_counted := WithLocal.handler_counted.
Abbreviation hd_error_combine := WithLocal.hd_error_combine.
Abbreviation heap_allocates := WithLocal.heap_allocates.
Abbreviation heap_ok := WithLocal.heap_ok.
Abbreviation heap_pa_ok := WithLocal.heap_pa_ok.
Abbreviation heap_pa_ok_le := WithLocal.heap_pa_ok_le.
Abbreviation indirect_ok := WithLocal.indirect_ok.
Abbreviation indirect_pending := WithLocal.indirect_pending.
Abbreviation initial_heap_ok := WithLocal.initial_heap_ok.
Abbreviation is_tagged_immediate_const := WithLocal.is_tagged_immediate_const.
Abbreviation length_unarize_app := WithLocal.length_unarize_app.
Abbreviation live := WithLocal.live.
Abbreviation load_field_nth := WithLocal.load_field_nth.
Abbreviation location_simple := WithLocal.location_simple.
Abbreviation lookup_counts := WithLocal.lookup_counts.
Abbreviation lookup_nodup_in := WithLocal.lookup_nodup_in.
Abbreviation may_come_from_any := WithLocal.may_come_from_any.
Abbreviation named_edges_no_heap := WithLocal.named_edges_no_heap.
Abbreviation named_edges_not_heap := WithLocal.named_edges_not_heap.
Abbreviation not_local := WithLocal.not_local.
Abbreviation nth_error_combine_eq := WithLocal.nth_error_combine_eq.
Abbreviation nth_error_combine_inv := WithLocal.nth_error_combine_inv.
Abbreviation nth_error_skipn_last := WithLocal.nth_error_skipn_last.
Abbreviation pending := WithLocal.pending.
Abbreviation prim_exec_pa_ok := WithLocal.prim_exec_pa_ok.
Abbreviation read_const := WithLocal.read_const.
Abbreviation read_location := WithLocal.read_location.
Abbreviation run_call_statics := WithLocal.run_call_statics.
Abbreviation simple_escapes := WithLocal.simple_escapes.
Abbreviation simple_flows := WithLocal.simple_flows.
Abbreviation simple_used := WithLocal.simple_used.
Abbreviation singleton_params_length := WithLocal.singleton_params_length.
Abbreviation skipn_length_app := WithLocal.skipn_length_app.
Abbreviation state_pa_ok := WithLocal.state_pa_ok.
Abbreviation statics_le_refl := WithLocal.statics_le_refl.
Abbreviation step_statics_le := WithLocal.step_statics_le.
Abbreviation store_field_replace := WithLocal.store_field_replace.
Abbreviation target_length := WithLocal.target_length.
Abbreviation transfers := WithLocal.transfers.
Abbreviation transfers_compose := WithLocal.transfers_compose.
Abbreviation tuple_fields_block := WithLocal.tuple_fields_block.
Abbreviation Conts_top := WithLocal.Conts_top.
Abbreviation Conts_code := WithLocal.Conts_code.
Abbreviation Conts_cons := WithLocal.Conts_cons.
Abbreviation Outer_ok := WithLocal.Outer_ok.
Abbreviation Closure_ok := WithLocal.Closure_ok.
Abbreviation Extra_ok := WithLocal.Extra_ok.
Abbreviation Over_ok := WithLocal.Over_ok.

Section Run.

Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable keep : value_slot -> bool.
Variable codes : list (code_id * code).
Variable symbols : list symbol.
Variable u : unit_interface.
Variable heap_prims : Prop.
Variable located : location -> option node.
Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hconsts : g_any_source G Node_all_constants.
Hypothesis Hsyms :
  ∀ s : symbol, noted s -> may_come_from sol (Node_symbol s) (Node_symbol s).
Hypothesis Hunit_syms :
  ∀ s : symbol,
    among s symbols = true -> may_come_from sol (Node_symbol s) (Node_symbol s).
Hypothesis Hdeps : ∀ d : closure_dep, deps d -> closure_dep_facts G cds d.
Hypothesis Hnd : NoDup (map fst cds).
Hypothesis Hcds :
  ∀ (id : code_id) (cd : code_dep),
    lookup id cds = Some cd -> known_witness_edges G id cd.

Abbreviation slots := (∀ f : field, is_local f = false).
Abbreviation no_local_reads :=
  (FlambdaReaperLocalFieldsProofs.local_reads_used_no_local is_local G sol).

Definition apply_conts :=
  @WithLocal.apply_conts is_local G sol noted deps cds keep codes symbols u
    heap_prims located slots Hc Hconsts Hsyms.
Definition apply_ok :=
  @WithLocal.apply_ok is_local G sol noted deps cds keep codes symbols u
    heap_prims located slots Hc Hlocal Hnd Hcds.
Definition call_ok :=
  @WithLocal.call_ok is_local G sol noted deps cds keep codes symbols u
    heap_prims located slots Hc Hlocal Hnd Hcds.
Definition call_step :=
  @WithLocal.call_step G noted deps cds keep codes symbols heap_prims slots.
Definition closure_call_ok :=
  @WithLocal.closure_call_ok is_local G sol noted deps cds keep codes symbols u
    heap_prims located slots Hc Hnd Hcds.
Definition code_facts :=
  @WithLocal.code_facts G noted deps cds keep codes symbols heap_prims slots
    Hnd.
Definition codes_ok :=
  @WithLocal.codes_ok G noted deps cds keep codes symbols heap_prims slots.
Definition cont_ok :=
  @WithLocal.cont_ok G sol noted deps cds keep codes symbols u heap_prims
    located slots.
Definition cont_ok_closure_inv :=
  @WithLocal.cont_ok_closure_inv G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition cont_ok_compose :=
  @WithLocal.cont_ok_compose G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition cont_ok_compose_gen :=
  @WithLocal.cont_ok_compose_gen G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition cont_ok_extra_inv :=
  @WithLocal.cont_ok_extra_inv G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition cont_ok_outer_inv :=
  @WithLocal.cont_ok_outer_inv G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition cont_ok_over_inv :=
  @WithLocal.cont_ok_over_inv G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition conts_ok :=
  @WithLocal.conts_ok G sol noted deps cds keep codes symbols u heap_prims
    located slots.
Definition conts_ok_flows :=
  @WithLocal.conts_ok_flows G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition conts_ok_lookup :=
  @WithLocal.conts_ok_lookup G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition conts_ok_none :=
  @WithLocal.conts_ok_none G sol noted deps cds keep codes symbols u heap_prims
    located slots.
Definition conts_ok_rec :=
  @WithLocal.conts_ok_rec G sol noted deps cds keep codes symbols u heap_prims
    located slots.
Definition direct_call_ok :=
  @WithLocal.direct_call_ok is_local G sol noted deps cds keep codes symbols u
    heap_prims located slots Hc Hlocal Hconsts Hsyms Hnd Hcds.
Definition enter_code_ok :=
  @WithLocal.enter_code_ok is_local G sol noted deps cds keep codes symbols u
    heap_prims located slots Hc Hnd.
Definition flow_invariant :=
  @WithLocal.flow_invariant G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition flow_invariant_flows :=
  @WithLocal.flow_invariant_flows G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition initial_flow_invariant :=
  @WithLocal.initial_flow_invariant G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition let_flow_invariant :=
  @WithLocal.let_flow_invariant is_local G sol noted deps cds keep codes
    symbols u heap_prims located slots Hc Hlocal Hconsts Hsyms Hdeps Hcds
    no_local_reads.
Definition named_flows :=
  @WithLocal.named_flows is_local G sol noted deps cds keep heap_prims located
    slots Hc Hlocal Hconsts Hsyms Hdeps Hcds no_local_reads.
Definition no_arg_call :=
  @WithLocal.no_arg_call G noted deps cds keep codes symbols heap_prims slots.
Definition prim_exec_flows :=
  @WithLocal.prim_exec_flows is_local G sol noted deps cds keep heap_prims
    located slots Hc Hlocal Hconsts Hsyms Hdeps Hcds no_local_reads.
Definition resolve_ok :=
  @WithLocal.resolve_ok G sol noted deps cds keep codes symbols u heap_prims
    located slots.
Definition result_heap_ok :=
  @WithLocal.result_heap_ok G sol noted deps cds keep codes symbols u
    heap_prims located slots.
Definition result_ok :=
  @WithLocal.result_ok G sol noted deps cds keep codes symbols u heap_prims
    located slots.
Definition run_flow_invariant :=
  @WithLocal.run_flow_invariant is_local G sol noted deps cds keep codes
    symbols u heap_prims located slots Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps
    Hnd Hcds no_local_reads.
Definition statics_bindings :=
  @WithLocal.statics_bindings is_local G sol noted deps cds keep codes symbols
    heap_prims located slots Hc Hconsts Hsyms Hunit_syms Hdeps Hcds.
Definition statics_bindings_pa :=
  @WithLocal.statics_bindings_pa G noted deps cds keep codes symbols heap_prims
    slots.
Definition step_flow_invariant :=
  @WithLocal.step_flow_invariant is_local G sol noted deps cds keep codes
    symbols u heap_prims located slots Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps
    Hnd Hcds no_local_reads.
Definition step_flow_invariant_core :=
  @WithLocal.step_flow_invariant_core is_local G sol noted deps cds keep codes
    symbols u heap_prims located slots Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps
    Hnd Hcds no_local_reads.
Definition step_flow_invariant_heap :=
  @WithLocal.step_flow_invariant_heap is_local G sol noted deps cds keep codes
    symbols u heap_prims located slots Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps
    Hnd Hcds no_local_reads.
Definition tupled_call_ok :=
  @WithLocal.tupled_call_ok is_local G sol noted deps cds keep codes symbols u
    heap_prims located slots Hc Hlocal Hnd Hcds.

End Run.

(* ======================================================================== *)
(* What the closures are said to have                                       *)
(* ======================================================================== *)


(* Where a value may be, with less said of the allocations of
   closures. *)
Lemma flows_weaken (G : graph) (sol : solution) (keep : value_slot -> bool)
    (called called' : node -> code_id -> Prop)
    (located : location -> option node) :
  (∀ (b : node) (id : code_id), called b id -> called' b id) ->
  ∀ (v : value) (n : node),
    flows G sol keep called located v n
    -> flows G sol keep called' located v n.
Proof.
  intros Hcalled v. pattern v.
  apply FlambdaHeapProofs.value_induction; clear v; simpl.
  - auto.
  - auto.
  - intros slot decls slots IH n (sites & Hn & Hall). exists sites.
    split; [ exact Hn | ]. intros f id Hf.
    destruct (Hall f id Hf) as (Hself & Hc & Hfs & Hvs).
    split; [ exact Hself | split; [ exact (Hcalled _ _ Hc) | ] ].
    split; [ exact Hfs | ].
    revert Hvs. apply keyed_for_all_bindings_impl.
    intros w field Hl H Hk. destruct (H Hk) as (from & Hcons & Hfrom).
    exists from. split; [ exact Hcons | ].
    rewrite Forall_forall in IH.
    exact (IH (w, field) (lookup_in _ _ _ Hl) from Hfrom).
  - auto.
  - intros tag fields IH n (a & Ha & His & Hgt & Hfields). exists a.
    split; [ exact Ha | split; [ exact His | split; [ exact Hgt | ] ] ].
    revert Hfields. apply fields_from_impl.
    intros i field Hin (from & Hcons & Hfrom). exists from.
    split; [ exact Hcons | ]. rewrite Forall_forall in IH.
    exact (IH field Hin from Hfrom).
  - intros elements IH n (Has & Hall). split; [ exact Has | ].
    revert Hall. apply every_impl.
    intros element Hin [Hconst | (m & Hm & Hu)]; [ left; exact Hconst | ].
    right. exists m. split; [ | exact Hu ]. rewrite Forall_forall in IH.
    exact (IH element Hin m Hm).
  - auto.
  - auto.
  - intros closure args a IHc IHargs n
      (m & ws & Hm & Hlen & Hchain & Hex & Hend & Hargs).
    exists m, ws. split; [ exact (IHc m Hm) | ].
    do 4 (split; [ assumption | ]).
    revert Hargs. apply fields_from_impl.
    intros i x Hin H g k w W p Hpos Hw HW Hp.
    rewrite Forall_forall in IHargs.
    exact (IHargs x Hin p (H g k w W p Hpos Hw HW Hp)).
Qed.

(* What a heap holds, with less said of the allocations of closures. *)
Lemma heap_flows_weaken (G : graph) (sol : solution)
    (keep : value_slot -> bool) (called called' : node -> code_id -> Prop)
    (located : location -> option node) (hp : heap) :
  (∀ (b : node) (id : code_id), called b id -> called' b id) ->
  heap_flows G sol keep called located hp ->
  heap_flows G sol keep called' located hp.
Proof.
  intros Hcalled H l c a Hl Ha. specialize (H l c a Hl Ha).
  destruct c as [mut tag fields | kind elements]; simpl in H |- *.
  - destruct H as (His & Hgt & Hfields).
    split; [ exact His | split; [ exact Hgt | ] ].
    revert Hfields. apply fields_from_impl.
    intros i field _ (from & Hcons & [Hf | (Hm & Hau & Hesc)]).
    + exists from. split; [ exact Hcons | left ].
      exact (flows_weaken _ _ _ _ _ _ Hcalled _ _ Hf).
    + exists from. split; [ exact Hcons | right ].
      split; [ exact Hm | split; [ exact Hau | ] ].
      destruct Hesc as [Hconst | (m & Hfm & Hau')]; [ left; exact Hconst | ].
      right. exists m.
      exact (conj (flows_weaken _ _ _ _ _ _ Hcalled _ _ Hfm) Hau').
  - revert H. apply every_impl.
    intros element _ [Hconst | (m & Hm & Hu)]; [ left; exact Hconst | ].
    right. exists m.
    exact (conj (flows_weaken _ _ _ _ _ _ Hcalled _ _ Hm) Hu).
Qed.

Lemma state_flows_weaken (G : graph) (sol : solution)
    (keep : value_slot -> bool) (called called' : node -> code_id -> Prop)
    (located : location -> option node) (s : state) :
  (∀ (b : node) (id : code_id), called b id -> called' b id) ->
  state_flows G sol keep called located s ->
  state_flows G sol keep called' located s.
Proof.
  intros Hcalled (Hvars & Hconts & Hst & Hheap).
  assert (Henv : ∀ vars : env,
            env_flows G sol keep called located vars ->
            env_flows G sol keep called' located vars).
  { intros vars H x v Hx. exact (flows_weaken _ _ _ _ _ _ Hcalled _ _ (H x v
    Hx)). }
  split; [ exact (Henv _ Hvars) | ].
  split; [ | split;
    [ | exact (heap_flows_weaken _ _ _ _ _ _ _ Hcalled Hheap) ] ].
  - intros k c Hin. specialize (Hconts k c Hin). clear Hin.
    induction Hconts as [h vars conts recursive Hv _ IH | k' | c extra _ IH
                        | remaining ra rta return_to exn_to _ IHret _ IHexn].
    + apply Closure_flows; [ exact (Henv _ Hv) | exact IH ].
    + apply Outer_flows.
    + apply Extra_args_flows. exact IH.
    + apply Over_application_flows; assumption.
  - intros s' allocated Hl. destruct (Hst s' allocated Hl) as [Hself Hrest].
    split; [ exact Hself | ].
    destruct allocated as [ | slot syms decls slots | tag fields | | | | ];
      try exact Hrest.
    + destruct Hrest as (Hc & Hfs & Hvs).
      split; [ intros id Hid; exact (Hcalled _ _ (Hc id Hid)) | ].
      split; [ exact Hfs | ].
      revert Hvs. apply keyed_for_all_bindings_impl.
      intros w field _ H Hk. destruct (H Hk) as (from & Hcons & Hfrom).
      exists from. split; [ exact Hcons | ].
      exact (flows_weaken _ _ _ _ _ _ Hcalled _ _ Hfrom).
    + destruct Hrest as (His & Hgt & Hfields).
      split; [ exact His | split; [ exact Hgt | ] ].
      revert Hfields. apply fields_from_impl.
      intros i field _ (from & Hcons & Hfrom). exists from.
      split;
        [ exact Hcons | exact (flows_weaken _ _ _ _ _ _ Hcalled _ _ Hfrom) ].
    + revert Hrest. apply fields_from_impl.
      intros i field _ (from & Hcons & Hfrom). exists from.
      split;
        [ exact Hcons | exact (flows_weaken _ _ _ _ _ _ Hcalled _ _ Hfrom) ].
Qed.

Lemma called_witness (G : graph) (cds : finite_map code_dep) (b : node)
    (id : code_id) :
  called_by G cds b id -> has_call_witness G b id.
Proof.
  unfold called_by, closure_dep_facts, has_call_witness. simpl.
  destruct (lookup id cds) as [cd | ].
  - intros (_ & H & _). eexists. exact H.
  - intros (w & _ & H). exists w. exact H.
Qed.

(* A node with any source is an allocation of its own. *)
Lemma may_come_from_self (is_local : field -> bool) (G : graph)
    (sol : solution) (a : node) :
  closed is_local G sol -> s_any_source sol a = true -> may_come_from sol a a.
Proof.
  intros Hc Has. split; [ | split; [ | split; [ | split ] ] ].
  - exact (as_hs _ _ _ Hc _ Has).
  - left. exact Has.
  - auto.
  - auto.
  - intros _ u' Hu. left. exact Hu.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* ======================================================================== *)
(* Without the two facts that keep a tuple alive                            *)
(* ======================================================================== *)

(* What the facts of a term read of a graph is in [G']: every fact but
   uses of other nodes than variables by other nodes than variables and
   symbols, and reads of [Is_int] into other nodes than variables, which
   are the two facts that keep the tuple of a call of tupled code
   alive. *)
Definition graph_sub (G G' : graph) : Prop :=
  (∀ a b : node, g_alias G a b -> g_alias G' a b)
  ∧ (∀ a b : node,
       g_use G a b ->
       (∃ x : variable, a = Node_variable x)
       ∨ (∃ s : symbol, a = Node_symbol s)
       ∨ (∃ y : variable, b = Node_variable y) ->
       g_use G' a b)
  ∧ (∀ (t : node) (f : field) (b : node),
       g_accessor G t f b ->
       f <> Field_is_int ∨ (∃ x : variable, t = Node_variable x) ->
       g_accessor G' t f b)
  ∧ (∀ (a : node) (f : field) (b : node),
       g_constructor G a f b -> g_constructor G' a f b)
  ∧ (∀ (a : node) (i : nat) (b : node),
       g_argument G a i b -> g_argument G' a i b)
  ∧ (∀ (a : node) (i : nat) (b : node),
       g_parameter G a i b -> g_parameter G' a i b)
  ∧ (∀ a b c : node, g_propagate G a b c -> g_propagate G' a b c)
  ∧ (∀ a : node, g_any_usage G a -> g_any_usage G' a)
  ∧ (∀ a : node, g_any_source G a -> g_any_source G' a).

Lemma graph_sub_refl (G : graph) : graph_sub G G.
Proof. repeat split; auto. Qed.

Lemma without_keep_alive_sub (F : facts) :
  graph_sub (graph_of_facts F)
    (graph_of_facts (FlambdaReaperTupledSpec.without_keep_alive F)).
Proof.
  unfold graph_sub, FlambdaReaperTupledSpec.without_keep_alive. cbn.
  split; [ auto | split; [ | split; [ | auto 10 ] ] ].
  - intros a b H Hcond. apply filter_In. split; [ exact H | ].
    destruct Hcond as [(x & ->) | [(s & ->) | (y & ->)]]; try reflexivity.
    destruct a; try reflexivity.
    all: simpl; apply negb_true_iff;
      unfold FlambdaReaperTupledSpec.keep_alive_nodes;
      induction (f_accessor F) as [ | [[t f] b'] l IH]; [ reflexivity | ];
      simpl; destruct t; try exact IH; destruct f; exact IH.
  - intros t f b H Hcond. apply filter_In. split; [ exact H | ].
    destruct Hcond as [Hf | (x & ->)]; [ | reflexivity ].
    destruct t; try reflexivity. destruct f; try reflexivity. contradiction.
Qed.

Section Mono.

Variables G G' : graph.
Hypothesis Hsub : graph_sub G G'.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable keep : value_slot -> bool.
Variable codes : list (code_id * code).
Variable symbols : list symbol.
Variable slots_read : Prop.
Variable heap_prims : Prop.

Lemma sub_alias (a b : node) : g_alias G a b -> g_alias G' a b.
Proof. apply Hsub. Qed.

Lemma sub_use_of_var (a : node) (y : variable) :
  g_use G a (Node_variable y) -> g_use G' a (Node_variable y).
Proof.
  intros H. apply (proj1 (proj2 Hsub)); [ exact H | right; right; eauto ].
Qed.

Lemma sub_use (x : variable) (b : node) :
  g_use G (Node_variable x) b -> g_use G' (Node_variable x) b.
Proof.
  intros H. apply (proj1 (proj2 Hsub)); [ exact H | left; eexists; eauto ].
Qed.

Lemma sub_use_symbol (s : symbol) (b : node) :
  g_use G (Node_symbol s) b -> g_use G' (Node_symbol s) b.
Proof.
  intros H. apply (proj1 (proj2 Hsub)); [ exact H | right; left; eauto ].
Qed.

Lemma sub_accessor (t : node) (f : field) (b : node) :
  f <> Field_is_int -> g_accessor G t f b -> g_accessor G' t f b.
Proof. intros Hf H. apply (proj1 (proj2 (proj2 Hsub)) t f b H). auto. Qed.

Lemma sub_accessor_var (x : variable) (f : field) (b : node) :
  g_accessor G (Node_variable x) f b -> g_accessor G' (Node_variable x) f b.
Proof.
  intros H. apply (proj1 (proj2 (proj2 Hsub)) _ f b H). right. eauto.
Qed.

Lemma sub_constructor (a : node) (f : field) (b : node) :
  g_constructor G a f b -> g_constructor G' a f b.
Proof. apply Hsub. Qed.

Lemma sub_argument (a : node) (i : nat) (b : node) :
  g_argument G a i b -> g_argument G' a i b.
Proof. apply Hsub. Qed.

Lemma sub_parameter (a : node) (i : nat) (b : node) :
  g_parameter G a i b -> g_parameter G' a i b.
Proof. apply Hsub. Qed.

Lemma sub_propagate (a b c : node) :
  g_propagate G a b c -> g_propagate G' a b c.
Proof. apply Hsub. Qed.

Lemma sub_any_usage (a : node) : g_any_usage G a -> g_any_usage G' a.
Proof. apply Hsub. Qed.

Lemma sub_any_source (a : node) : g_any_source G a -> g_any_source G' a.
Proof. apply Hsub. Qed.

Ltac not_is_int := discriminate.

Lemma block_edges_mono (base : node) (fields : list simple) :
  block_edges G noted base fields -> block_edges G' noted base fields.
Proof.
  intros (Hf & Hi & Hg & Hn).
  split; [ intros i s H; exact (sub_constructor _ _ _ (Hf i s H)) | ].
  split; [ exact (sub_constructor _ _ _ Hi) | ].
  split; [ exact (sub_constructor _ _ _ Hg) | exact Hn ].
Qed.

Lemma boxed_edges_mono (s : symbol) (contents : simple) :
  boxed_edges G s contents -> boxed_edges G' s contents.
Proof.
  unfold boxed_edges. destruct (boxed_number_kind contents);
    [ apply sub_constructor | ].
  intros [Hal Huse]. split; [ exact (sub_alias _ _ Hal) | ].
  intros x Hx. exact (sub_use_symbol _ _ (Huse x Hx)).
Qed.

Lemma accessor_edges_mono (x : variable) (f : field) (s : simple) :
  accessor_edges G noted x f s -> accessor_edges G' noted x f s.
Proof. intros [H Hn]. exact (conj (sub_accessor_var _ _ _ H) Hn). Qed.

Lemma prim_edges_mono (ctx : option code_id) (x : variable) (p : prim)
    (args : list simple) :
  prim_edges G noted heap_prims ctx x p args ->
  prim_edges G' noted heap_prims ctx x p args.
Proof.
  unfold prim_edges. intros [(Hoff & Heff & Hmut) H].
  split; [ split; [ exact Hoff | split ] | revert H ].
  { intros He. specialize (Heff He). unfold cond_use_edge in Heff |- *.
    destruct ctx;
      [ exact (sub_use_of_var _ _ Heff) | exact (sub_any_usage _ Heff) ]. }
  { intros Hm. exact (sub_any_source _ (Hmut Hm)). }
  destruct (prim_classified p args).
  - unfold classified_prim_edges.
    repeat match goal with
           | |- context [match ?t with _ => _ end] => destruct t
           end;
      first [ exact (fun H => H) | apply block_edges_mono
            | apply accessor_edges_mono
            | intros [Hc Hn]; exact (conj (sub_constructor _ _ _ Hc) Hn) ].
  - intros [Has Huse]. split; [ exact (sub_any_source _ Has) | ].
    intros from Hin. exact (sub_use _ _ (Huse from Hin)).
Qed.

Lemma closure_edges_mono (b : node) (id : code_id)
    (function_decls : list (function_slot * code_id))
    (sites : function_slot -> node) (value_slots : list (value_slot * simple)) :
  closure_edges G noted deps keep b id function_decls sites value_slots ->
  closure_edges G' noted deps keep b id function_decls sites value_slots.
Proof.
  intros (Hdep & Hfs & Hvs). split; [ exact Hdep | split ].
  - intros g id' Hg. exact (sub_constructor _ _ _ (Hfs g id' Hg)).
  - intros w s Hw Hk. destruct (Hvs w s Hw Hk) as [Hc Hn].
    exact (conj (sub_constructor _ _ _ Hc) Hn).
Qed.

Lemma named_edges_mono (ctx : option code_id) (x : variable) (d : named) :
  named_edges G noted deps keep slots_read heap_prims ctx x d ->
  named_edges G' noted deps keep slots_read heap_prims ctx x d.
Proof.
  destruct d as [s | p args | code vs | slot decls vs | slot closure
                | from w closure | from to_ closure]; simpl.
  - intros [H Hn]. exact (conj (sub_alias _ _ H) Hn).
  - apply prim_edges_mono.
  - auto.
  - intros H id0 Hid0. destruct (H id0 Hid0) as (sites & Hx & Hall).
    exists sites. split; [ exact Hx | ].
    intros f id Hf. exact (closure_edges_mono _ _ _ _ _ (Hall f id Hf)).
  - auto.
  - intros (Ha & Hk & Hs). exact (conj (accessor_edges_mono _ _ _ Ha)
                                    (conj Hk Hs)).
  - intros (Ha & Hs). exact (conj (accessor_edges_mono _ _ _ Ha) Hs).
Qed.

Lemma static_set_edges_mono (s : symbol) (slot : function_slot)
    (closure_symbols : list (function_slot * symbol))
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) :
  static_set_edges G noted deps keep symbols s slot closure_symbols
    function_decls value_slots ->
  static_set_edges G' noted deps keep symbols s slot closure_symbols
    function_decls value_slots.
Proof.
  intros (Hb & Hdep & Hfs & Hvs).
  split; [ exact Hb | split; [ exact Hdep | split ] ].
  - intros g s' Hg. destruct (Hfs g s' Hg) as [Hc Ha].
    exact (conj (sub_constructor _ _ _ Hc) Ha).
  - intros w x Hw Hk. destruct (Hvs w x Hw Hk) as [Hc Hn].
    exact (conj (sub_constructor _ _ _ Hc) Hn).
Qed.

Lemma known_witness_edges_mono (id : code_id) (cd : code_dep) :
  known_witness_edges G id cd -> known_witness_edges G' id cd.
Proof.
  intros (Hp & Hr & He & Hc & Has).
  split; [ intros i p H; exact (sub_parameter _ _ _ (Hp i p H)) | ].
  split; [ intros i r H; exact (sub_constructor _ _ _ (Hr i r H)) | ].
  split; [ exact (sub_constructor _ _ _ He) | ].
  split; [ exact (sub_constructor _ _ _ Hc) | exact (sub_any_source _ Has) ].
Qed.

Lemma tupled_witness_edges_mono (id : code_id) (cd : code_dep) :
  tupled_witness_edges G id cd -> tupled_witness_edges G' id cd.
Proof.
  intros (w & U & Heq & Hr & He & Hc & Hp & Hacc).
  exists w, U. split; [ exact Heq | ].
  split; [ intros i r H; exact (sub_constructor _ _ _ (Hr i r H)) | ].
  split; [ exact (sub_constructor _ _ _ He) | ].
  split; [ exact (sub_constructor _ _ _ Hc) | ].
  split; [ exact (sub_parameter _ _ _ Hp) | ].
  intros i p H. apply sub_accessor; [ not_is_int | exact (Hacc i p H) ].
Qed.

Lemma code_chain_edges_mono (id : code_id) (returns : list node) (exn : node)
    (L : list (list node * node)) :
  code_chain_edges G id returns exn L -> code_chain_edges G' id returns exn L.
Proof.
  induction L as [ | [first w] L IH]; [ auto | ].
  intros (Hp & Hc & Hnext & HL).
  split; [ intros k p H; exact (sub_parameter _ _ _ (Hp k p H)) | ].
  split; [ exact (sub_constructor _ _ _ Hc) | split; [ | exact (IH HL) ] ].
  destruct L as [ | [first' w'] L'].
  - destruct Hnext as [He Hr]. split; [ exact (sub_constructor _ _ _ He) | ].
    intros i r H. exact (sub_constructor _ _ _ (Hr i r H)).
  - destruct Hnext as (v & Hv & Hvw). exists v.
    exact (conj (sub_constructor _ _ _ Hv) (sub_constructor _ _ _ Hvw)).
Qed.

Lemma caller_chain_edges_mono (returns : list node) (exn called : node)
    (L : list (list simple * node)) :
  caller_chain_edges G returns exn called L ->
  caller_chain_edges G' returns exn called L.
Proof.
  induction L as [ | [first w] L IH]; [ auto | ].
  intros (Ha & He & Hc & Hnext & HL).
  split; [ intros k s H; exact (sub_argument _ _ _ (Ha k s H)) | ].
  split; [ apply sub_accessor; [ not_is_int | exact He ] | ].
  split; [ apply sub_accessor; [ not_is_int | exact Hc ] | ].
  split; [ | exact (IH HL) ].
  destruct L as [ | [first' w'] L'].
  - intros i r H. apply sub_accessor; [ not_is_int | exact (Hnext i r H) ].
  - destruct Hnext as (v & Hv & Hvw). exists v.
    split; (apply sub_accessor; [ not_is_int | assumption ]).
Qed.

Lemma cond_alias_edge_mono (ctx : option code_id) (to_ from : node) :
  cond_alias_edge G ctx to_ from -> cond_alias_edge G' ctx to_ from.
Proof.
  destruct ctx; simpl; [ apply sub_propagate | apply sub_alias ].
Qed.

Lemma apply_cont_edges_mono (ce : finite_map (list node)) (ac : apply_cont) :
  apply_cont_edges G noted ce ac -> apply_cont_edges G' noted ce ac.
Proof.
  intros [Ha Hn]. split; [ | exact Hn ].
  intros params Hp p s Hin. exact (sub_alias _ _ (Ha params Hp p s Hin)).
Qed.

Lemma apply_edges_mono (ctx : option code_id) (ce : finite_map (list node))
    (ap : apply_expr) :
  apply_edges G noted codes ctx ce ap -> apply_edges G' noted codes ctx ce ap.
Proof.
  intros (Hfrag & Hnc & Hna & returns & exn & Hret & Hexn & Hcall).
  split; [ exact Hfrag | split; [ exact Hnc | split; [ exact Hna | ] ] ].
  exists returns, exn. split; [ exact Hret | split ].
  - intros p params H. destruct (Hexn p params H) as (Heq & Hn & Ha).
    split; [ exact Heq | split; [ exact Hn | ] ].
    intros q s Hin. exact (sub_alias _ _ (Ha q s Hin)).
  - unfold call_edges in Hcall |- *.
    destruct (ap_call_kind ap) as [ | | ];
      [ | | destruct Hcall as (witness & apply_node & called_node & Hargs
                               & Hrets & Hexnacc & Hcalled & Hused & Hcond
                               & Happly);
            exists witness, apply_node, called_node;
            split; [ intros i s H; exact (sub_argument _ _ _ (Hargs i s H))
                   | ];
            split; [ intros i r H; apply sub_accessor;
                     [ not_is_int | exact (Hrets i r H) ] | ];
            split; [ apply sub_accessor; [ not_is_int | exact Hexnacc ] | ];
            split; [ apply sub_accessor; [ not_is_int | exact Hcalled ] | ];
            split; [ exact (sub_any_usage _ Hused) | ];
            split; [ exact (cond_alias_edge_mono _ _ _ Hcond) | ];
            apply sub_accessor; [ not_is_int | exact Happly ] ].
    + destruct Hcall as (witness & apply_node & called_node & Hargs & Hrets
                         & Hexnacc & Hcalled & Hused & Hcond & Happly).
      exists witness, apply_node, called_node.
      split; [ intros i s H; exact (sub_argument _ _ _ (Hargs i s H)) | ].
      split; [ intros i r H;
               apply sub_accessor; [ not_is_int | exact (Hrets i r H) ] | ].
      split; [ apply sub_accessor; [ not_is_int | exact Hexnacc ] | ].
      split; [ apply sub_accessor; [ not_is_int | exact Hcalled ] | ].
      split; [ exact (sub_any_usage _ Hused) | ].
      split; [ exact (cond_alias_edge_mono _ _ _ Hcond) | ].
      apply sub_accessor; [ not_is_int | exact Happly ].
    + destruct Hcall as (ws & apply_node & called_node & Hlen & Hch & Hused
                         & Hcond & Happly).
      exists ws, apply_node, called_node.
      split; [ exact Hlen | split;
        [ exact (caller_chain_edges_mono _ _ _ _ Hch) | ] ].
      split; [ exact (sub_any_usage _ Hused) | split ].
      * intros w Hw. exact (cond_alias_edge_mono _ _ _ (Hcond w Hw)).
      * apply sub_accessor; [ not_is_int | exact Happly ].
Qed.

Lemma expr_edges_mono :
  (∀ (e : expr) (ctx : option code_id) (ce : finite_map (list node)),
     expr_edges G noted deps cds keep codes symbols slots_read heap_prims ctx ce
       e ->
     expr_edges G' noted deps cds keep codes symbols slots_read heap_prims ctx
       ce e)
  ∧ (∀ (h : continuation_handler) (ctx : option code_id)
       (ce : finite_map (list node)),
       handler_edges G noted deps cds keep codes symbols slots_read heap_prims
         ctx ce h ->
       handler_edges G' noted deps cds keep codes symbols slots_read heap_prims
         ctx ce h).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH ctx ce [Hd Hbody]. simpl.
    exact (conj (named_edges_mono _ _ _ Hd) (IH _ _ Hbody)).
  - intros bindings body IHcode IH ctx ce [Hbindings Hbody]. simpl.
    split; [ | exact (IH _ _ Hbody) ].
    revert Hbindings. apply every_impl. intros b Hin Hb.
    destruct b as [id c | id | s const]; simpl in Hb |- *.
    + destruct Hb as [Hin_codes Hcode]. split; [ exact Hin_codes | ].
      specialize (IHcode id c Hin).
      destruct c as [params mc ret exn metadata cbody]. simpl in IHcode.
      destruct Hcode as (cd & Hcd & Hkw & Hunk & Htw & Lp & Lr & Hmc & Hps
                         & Hne & Hbinds & Hcounted & Hbody').
      exists cd. split; [ exact Hcd | ].
      split; [ exact (known_witness_edges_mono _ _ Hkw) | ].
      split; [ intros Ht; destruct (Hunk Ht) as [Lu Hch];
               exact (conj Lu (code_chain_edges_mono _ _ _ _ Hch)) | ].
      split; [ intros Ht; exact (tupled_witness_edges_mono _ _ (Htw Ht)) | ].
      split; [ exact Lp | split; [ exact Lr | ] ].
      split; [ exact (sub_alias _ _ Hmc) | ].
      split; [ intros p q H; exact (sub_alias _ _ (Hps p q H)) | ].
      split; [ exact Hne | split; [ exact Hbinds | ] ].
      split; [ exact Hcounted | exact (IHcode _ _ Hbody') ].
    + exact Hb.
    + destruct const as [ | slot syms decls vs | t fields | | | ];
        try exact Hb.
      * exact (static_set_edges_mono _ _ _ _ _ Hb).
      * exact (block_edges_mono _ _ Hb).
      * exact (boxed_edges_mono _ _ Hb).
      * exact (sub_alias _ _ Hb).
      * destruct elements; [ exact (sub_alias _ _ Hb) | ].
        exact (block_edges_mono _ _ Hb).
  - intros k h body IHh IH ctx ce [Hh Hbody]. simpl.
    exact (conj (IHh _ _ Hh) (IH _ _ Hbody)).
  - intros handlers body IHh IH ctx ce [Hhandlers Hbody]. simpl.
    split; [ | exact (IH _ _ Hbody) ].
    revert Hhandlers. apply every_impl. intros [k h] Hin Hh.
    exact (IHh k h Hin _ _ Hh).
  - intros ap ctx ce H. exact (apply_edges_mono _ _ _ H).
  - intros ac ctx ce H. exact (apply_cont_edges_mono _ _ H).
  - intros scrutinee arms ctx ce H. simpl in H |- *.
    revert H. apply every_impl. intros [i ac] _ H.
    exact (apply_cont_edges_mono _ _ H).
  - intros ctx ce H. exact H.
  - intros params metadata handler IH ctx ce H. exact (IH _ _ H).
Qed.

Lemma closure_dep_facts_mono (d : closure_dep) :
  closure_dep_facts G cds d -> closure_dep_facts G' cds d.
Proof.
  unfold closure_dep_facts.
  destruct (lookup (cld_closure_code_id d) cds) as [cd | ].
  - intros (Hp & Hk & Hu).
    split; [ exact (sub_propagate _ _ _ Hp) | ].
    split; [ exact (sub_constructor _ _ _ Hk) | ].
    intros w Hw. exact (sub_constructor _ _ _ (Hu w Hw)).
  - intros (w & Has & Hk). exists w.
    exact (conj (sub_any_source _ Has) (sub_constructor _ _ _ Hk)).
Qed.

End Mono.

(* The statement, of every graph that has the facts that the proof reads
   of the graph of the unit. *)
(* ======================================================================== *)
(* The allocations of the locations of a run                                *)
(* ======================================================================== *)

(* The variable of the [Let] that a state allocates with, if it does. *)
Definition allocating_var (s : state) : option variable :=
  match state_expr s with
  | Let x _ (Prim (Heap q) _) _ => if heap_allocates q then Some x else None
  | _ => None
  end.

(* A step that allocates binds the fresh location. *)
Lemma allocating_binds (s s1 : state) (x : variable) :
  allocating_var s = Some x -> step s = Step s1 ->
  lookup (fresh_location (state_heap s)) (state_heap s1) <> None.
Proof.
  destruct s as [e vars conts statics hp rd]. unfold allocating_var.
  simpl_state.
  destruct e as [y kind d body | | | | | | | ]; try discriminate.
  destruct d as [ | p args | | | | | ]; try discriminate.
  destruct p as [ | | | q | ]; try discriminate.
  destruct (heap_allocates q) eqn:Hq; [ intros [= <-] | discriminate ].
  rewrite step_let_exec.
  destruct (exec_named statics hp rd vars (Prim (Heap q) args))
    as [[[v hp'] rd'] | ] eqn:Hexec; [ | discriminate ].
  destruct (kind_eqb _ _); [ intros [= <-] | discriminate ]. simpl_state.
  apply exec_named_prim_some in Hexec as (vs & _ & Hprim & _).
  unfold exec_prim, exec_bound_prim in Hprim.
  destruct (forallb _ _); [ | discriminate Hprim ].
  destruct q; try discriminate Hq; cbn [exec_heap_primitive] in Hprim.
  - destruct (forallb _ _); [ | discriminate Hprim ].
    injection Hprim as _ <-. rewrite lookup_cons, Pos.eqb_refl. discriminate.
  - destruct vs as [ | v0 vs0]; [ discriminate Hprim | ].
    destruct (array_made _ _); simpl in Hprim; [ | discriminate Hprim ].
    injection Hprim as _ <-. rewrite lookup_cons, Pos.eqb_refl. discriminate.
Qed.

Lemma heap_evolves_bound (h h' : heap) (l : location) :
  heap_evolves h h' -> lookup l h <> None -> lookup l h' <> None.
Proof.
  intros [Hb Ha] Hl.
  destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:E;
    [ | | contradiction ].
  - destruct (Hb l mut tag fields E) as (fields' & H' & _). rewrite H'.
    discriminate.
  - destruct (Ha l kind elements E) as (elements' & H' & _). rewrite H'.
    discriminate.
Qed.

(* A location bound stays bound along a run. *)
Lemma run_keeps_bound (i : nat) (s s' : state) (l : location) :
  run i s = Step s' -> lookup l (state_heap s) <> None ->
  lookup l (state_heap s') <> None.
Proof.
  revert s. induction i as [ | i IH]; intros s Hrun Hl.
  - simpl in Hrun. injection Hrun as <-. exact Hl.
  - rewrite run_succ in Hrun.
    destruct (step s) as [s1 | | ] eqn:E; try discriminate Hrun.
    apply (IH s1 Hrun).
    exact (heap_evolves_bound _ _ _ (step_evolves s s1 E) Hl).
Qed.

(* The first [n] steps of a run allocate each location once, at the
   [Let] of a variable, whose node the location is given. *)
Lemma located_exists (n : nat) (s : state) :
  ∃ located : location -> option node,
    ∀ (i : nat) (s' : state), i < n -> run i s = Step s' ->
      alloc_ok located s'.
Proof.
  revert s. induction n as [ | n IH]; intros s.
  - exists (fun _ => None). intros i s' Hi. lia.
  - set (L0 := fun (L : location -> option node) (l : location) =>
                 match allocating_var s with
                 | Some x =>
                   if Pos.eqb l (fresh_location (state_heap s))
                   then Some (Node_variable x) else L l
                 | None => L l
                 end).
    assert (H0 : ∀ L : location -> option node, alloc_ok (L0 L) s).
    { intros L x kind q args body Hexpr Hq. unfold L0, allocating_var.
      rewrite Hexpr, Hq, Pos.eqb_refl. reflexivity. }
    destruct (step s) as [s1 | k vs hp rd | ] eqn:E.
    + destruct (IH s1) as (L1 & HL1).
      exists (L0 L1). intros [ | i] s' Hi Hrun.
      * simpl in Hrun. injection Hrun as <-. apply H0.
      * rewrite run_succ, E in Hrun.
        pose proof (HL1 i s' ltac:(lia) Hrun) as Hok.
        intros x kind q args body Hexpr Hq.
        rewrite <- (Hok x kind q args body Hexpr Hq).
        unfold L0. destruct (allocating_var s) as [y | ] eqn:Hy;
          [ | reflexivity ].
        destruct (Pos.eqb_spec (fresh_location (state_heap s'))
                    (fresh_location (state_heap s))) as [Heq | _];
          [ | reflexivity ].
        exfalso.
        pose proof (run_keeps_bound i s1 s' _ Hrun
                      (allocating_binds s s1 y Hy E)) as Hb.
        rewrite <- Heq in Hb.
        exact (Hb (FlambdaHeapProofs.fresh_location _)).
    + exists (L0 (fun _ => None)). intros [ | i] s' Hi Hrun.
      * simpl in Hrun. injection Hrun as <-. apply H0.
      * rewrite run_succ, E in Hrun. discriminate Hrun.
    + exists (L0 (fun _ => None)). intros [ | i] s' Hi Hrun.
      * simpl in Hrun. injection Hrun as <-. apply H0.
      * rewrite run_succ, E in Hrun. discriminate Hrun.
Qed.

(* ======================================================================== *)
(* The reads of a unit without projection                                   *)
(* ======================================================================== *)

(* A fact that reads no value or function slot. *)
Definition reads_no_slot (fct : fact) : bool :=
  match fct with
  | Fact_accessor _ (Field_value_slot _) _
  | Fact_accessor _ (Field_function_slot _) _ => false
  | _ => true
  end.

Section NoSlotReads.

Abbreviation nsr l := (forallb reads_no_slot l = true).
Abbreviation nsa a := (nsr (acc_facts a)).

Lemma nsr_app (l l' : list fact) : nsr l -> nsr l' -> nsr (l ++ l').
Proof. intros H H'. rewrite forallb_app, H, H'. reflexivity. Qed.

Lemma nsr_map {A : Type} (g : A -> fact) (l : list A) :
  (∀ x : A, reads_no_slot (g x) = true) -> nsr (map g l).
Proof.
  intros H. induction l as [ | x l IH]; [ reflexivity | ].
  simpl. rewrite H. exact IH.
Qed.

Lemma nsa_add_facts (l : list fact) (a : acc) :
  nsr l -> nsa a -> nsa (add_facts l a).
Proof. intros Hl Ha. exact (nsr_app _ _ Hl Ha). Qed.

Lemma note_simple_facts_eq (s : simple) (a : acc) :
  acc_facts (note_simple s a) = acc_facts a.
Proof. destruct s; reflexivity. Qed.

Lemma note_simples_facts_eq (l : list simple) :
  ∀ a : acc, acc_facts (note_simples l a) = acc_facts a.
Proof.
  unfold note_simples. induction l as [ | s l IH]; intros a;
    [ reflexivity | ].
  simpl. rewrite IH. apply note_simple_facts_eq.
Qed.

Lemma fresh_list_facts_eq (n : nat) :
  ∀ a : acc, acc_facts (snd (fresh_list n a)) = acc_facts a.
Proof.
  induction n as [ | n IH]; intros a; [ reflexivity | ].
  cbn [fresh_list]. destruct (fresh a) as [x a1] eqn:Ef.
  specialize (IH a1). destruct (fresh_list n a1) as [xs a2].
  cbn [snd] in IH |- *. rewrite IH.
  unfold fresh in Ef. injection Ef as _ <-. reflexivity.
Qed.

Ltac nsr_tac :=
  repeat first
    [ reflexivity
    | apply nsr_app
    | apply nsr_map; intros [? ?]; reflexivity
    | apply nsr_map; intros ?; reflexivity ].

Lemma nsa_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  nsa a -> nsa (cond_any_usage env s a).
Proof.
  intros Ha. unfold cond_any_usage.
  destruct (env_current_code_id env); apply nsa_add_facts;
    try reflexivity; rewrite note_simple_facts_eq; exact Ha.
Qed.

Lemma nsa_cond_alias (env : traverse_env) (from to_ : node) (a : acc) :
  nsa a -> nsa (cond_alias env from to_ a).
Proof.
  intros Ha. unfold cond_alias.
  destruct (env_current_code_id env); apply nsa_add_facts;
    first [ reflexivity | exact Ha ].
Qed.

Lemma nsa_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  nsa a -> nsa (snd (find_cont env k n a)).
Proof.
  intros Ha. unfold find_cont.
  destruct (lookup k (FlambdaReaperTraversal.env_conts env)); [ exact Ha | ].
  pose proof (fresh_list_facts_eq n a) as Hf.
  destruct (fresh_list n a) as [params a']. cbn [snd] in Hf |- *.
  apply nsa_add_facts; [ nsr_tac | rewrite Hf; exact Ha ].
Qed.

Lemma nsa_block_deps (base : node) (fields : list simple) (a : acc) :
  nsa a -> nsa (block_deps base fields a).
Proof.
  intros Ha. unfold block_deps. apply nsa_add_facts; [ nsr_tac | ].
  rewrite note_simples_facts_eq. exact Ha.
Qed.

Lemma nsa_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  nsa a -> nsa (traverse_prim env x p args a).
Proof.
  intros Ha. unfold traverse_prim.
  assert (Hacc : ∀ f s a,
            reads_no_slot (Fact_accessor (Node_variable x) f
                             (node_of_simple s)) = true ->
            nsa a ->
            nsa (add_fact (Fact_accessor (Node_variable x) f
                             (node_of_simple s)) (note_simple s a))).
  { intros f s a0 Hf Ha0. apply nsa_add_facts.
    - exact (andb_true_intro (conj Hf eq_refl)).
    - rewrite note_simple_facts_eq. exact Ha0. }
  assert (Hdefault :
    nsa (add_facts
           (Fact_any_source (Node_variable x)
            :: map (fun from => Fact_use (Node_variable x) from)
                 (free_names_of_simples args))
           (if prim_has_arbitrary_effects p
            then cond_any_usage env (Var x) a else a))).
  { apply nsa_add_facts; [ nsr_tac | ].
    destruct (prim_has_arbitrary_effects p);
      [ apply nsa_cond_any_usage, Ha | exact Ha ]. }
  cbv zeta.
  destruct p as [p | p | p | p | p]; [ | exact Hdefault | | | ].
  - destruct p; destruct args as [ | s [ | s' args]]; try exact Hdefault;
      try destruct mut; try destruct variant_only; cbv beta iota;
      first
        [ exact Hdefault
        | apply Hacc; [ reflexivity | exact Ha ]
        | apply nsa_add_facts; [ reflexivity | ];
          first
            [ apply Hacc; [ reflexivity | exact Ha ]
            | rewrite note_simple_facts_eq; exact Ha ] ].
  - destruct p; [ apply nsa_block_deps, Ha | exact Hdefault ].
  - destruct p; try exact Hdefault. apply nsa_block_deps, Ha.
  - destruct p; try exact Hdefault. destruct args; exact Ha.
Qed.

Lemma nsa_record_set_of_closures_deps (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  nsa a ->
  nsa (record_set_of_closures_deps env closures function_slots value_slots
         a).
Proof.
  unfold record_set_of_closures_deps. revert a.
  induction closures as [ | [base code] closures IH]; intros a Ha;
    [ exact Ha | ].
  cbn [fold_left]. apply IH. apply nsa_add_facts; [ nsr_tac | ].
  rewrite note_simples_facts_eq. destruct code; exact Ha.
Qed.

Lemma closures_of_set_facts (x : variable) (slot : function_slot)
    (others : list (function_slot * variable))
    (function_decls : list (function_slot * code_id)) :
  ∀ a : acc,
    acc_facts (snd (closures_of_set x slot others function_decls a))
    = acc_facts a.
Proof.
  induction function_decls as [ | [f id] decls IH]; intros a;
    [ reflexivity | ].
  cbn [closures_of_set].
  destruct (if Pos.eqb f slot then (Node_variable x, a)
            else match lookup f others with
                 | Some y => (Node_variable y, a)
                 | None => fresh a
                 end) as [name a1] eqn:Hn.
  specialize (IH a1).
  destruct (closures_of_set x slot others decls a1) as [rest a2].
  cbn [snd] in IH |- *. rewrite IH.
  destruct (Pos.eqb f slot); [ injection Hn as _ <-; reflexivity | ].
  destruct (lookup f others); [ injection Hn as _ <-; reflexivity | ].
  unfold fresh in Hn. injection Hn as _ <-. reflexivity.
Qed.

Lemma nsa_traverse_closure_of_set (env : traverse_env) (x : variable)
    (slot : function_slot) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (a : acc) :
  nsa a -> nsa (traverse_closure_of_set env x slot function_decls value_slots
                  a).
Proof.
  intros Ha. unfold traverse_closure_of_set.
  destruct (among slot _); [ | exact Ha ].
  pose proof (closures_of_set_facts x slot (set_closures_of env x)
                (compact function_decls) a) as Hf.
  destruct (closures_of_set _ _ _ _ a) as [closures a1].
  cbn [snd] in Hf. apply nsa_record_set_of_closures_deps.
  rewrite Hf. exact Ha.
Qed.

Lemma nsa_traverse_named (env : traverse_env) (x : variable) (d : named)
    (a : acc) :
  named_projection_free d = true ->
  nsa a -> nsa (traverse_named env x d a).
Proof.
  intros Hd Ha. destruct d; try discriminate Hd; cbn [traverse_named].
  - apply nsa_add_facts; [ reflexivity | ].
    rewrite note_simple_facts_eq. exact Ha.
  - apply nsa_traverse_prim, Ha.
  - apply nsa_record_set_of_closures_deps, Ha.
  - apply nsa_traverse_closure_of_set, Ha.
Qed.

Lemma nsa_traverse_static_binding (env : traverse_env)
    (binding : static_binding) (a : acc) :
  nsa a -> nsa (traverse_static_binding env binding a).
Proof.
  intros Ha. destruct binding as [ | | s const]; try exact Ha.
  destruct const as [ | | | contents | | elements ];
    cbn [traverse_static_binding].
  - apply nsa_record_set_of_closures_deps, Ha.
  - apply nsa_record_set_of_closures_deps, Ha.
  - apply nsa_block_deps, Ha.
  - apply nsa_add_facts; [ | exact Ha ].
    unfold boxed_number_facts.
    destruct (boxed_number_kind contents); [ reflexivity | ].
    destruct contents; reflexivity.
  - apply nsa_add_facts; [ reflexivity | exact Ha ].
  - destruct elements; [ apply nsa_add_facts; [ reflexivity | exact Ha ] | ].
    apply nsa_block_deps, Ha.
Qed.

Lemma nsa_unknown_arity_witness_deps (id : code_id) (returns : list node)
    (exn : node) :
  ∀ (pws : list (list node * node)) (a : acc),
    nsa a -> nsa (unknown_arity_witness_deps id returns exn pws a).
Proof.
  induction pws as [ | [first witness] rest IH]; intros a Ha;
    [ exact Ha | ].
  cbn [unknown_arity_witness_deps].
  assert (Ha1 : nsa (add_facts
                       (map (fun '(i, p) => Fact_parameter witness i p)
                          (indexed first)
                        ++ [Fact_constructor witness
                              Field_code_id_of_call_witness
                              (Node_code_id id)]) a))
    by (apply nsa_add_facts; [ nsr_tac | exact Ha ]).
  destruct rest as [ | [first' next] rest'].
  - apply nsa_add_facts; [ nsr_tac | exact Ha1 ].
  - destruct (fresh _) as [v a2] eqn:Hf. apply IH.
    apply nsa_add_facts; [ reflexivity | ].
    unfold fresh in Hf. injection Hf as _ <-. exact Ha1.
Qed.

Lemma nsa_prepare_code (id : code_id) (c : code) (a : acc) :
  nsa a -> nsa (prepare_code id c a).
Proof.
  intros Ha. unfold prepare_code.
  pose proof (fresh_list_facts_eq (length (code_result_kinds c)) a) as H1.
  destruct (fresh_list _ a) as [returns a1]. cbn [snd] in H1.
  destruct (fresh a1) as [exn a2] eqn:H2.
  destruct (fresh a2) as [my_closure a3] eqn:H3.
  pose proof (fresh_list_facts_eq (length (code_params_kinds c)) a3) as H4.
  destruct (fresh_list _ a3) as [params a4]. cbn [snd] in H4.
  unfold fresh in H2, H3. injection H2 as _ <-. injection H3 as _ <-.
  cbn [acc_facts] in H4.
  assert (Ha4 : nsa a4) by (rewrite H4, H1; exact Ha).
  unfold create_known_arity_call_witness.
  destruct (fresh a4) as [known a5] eqn:H5.
  unfold fresh in H5. injection H5 as _ <-.
  set (a6 := add_facts _ _).
  assert (Ha6 : nsa a6) by (apply nsa_add_facts; [ nsr_tac | exact Ha4 ]).
  clearbody a6.
  assert (Hu : nsa (snd (create_unknown_arity_call_witnesses id
                           (code_is_tupled c) (code_params_arity c) params
                           returns exn a6))).
  { unfold create_unknown_arity_call_witnesses.
    destruct (code_is_tupled c).
    - unfold create_unknown_arity_tupled_call_witnesses.
      destruct (fresh a6) as [w b1] eqn:E1.
      destruct (fresh b1) as [ut b2] eqn:E2.
      destruct (fresh b2) as [ka b3] eqn:E3.
      unfold fresh in E1, E2, E3.
      injection E1 as _ <-. injection E2 as _ <-. injection E3 as _ <-.
      apply nsa_add_facts; [ nsr_tac | exact Ha6 ].
    - unfold create_unknown_arity_non_tupled_call_witnesses.
      pose proof (fresh_list_facts_eq
                    (length (at_least_one_group
                               (group_by_parameter (code_params_arity c)
                                  params))) a6) as Hf.
      destruct (fresh_list _ a6) as [ws b1]. cbn [snd] in Hf |- *.
      apply nsa_unknown_arity_witness_deps. rewrite Hf. exact Ha6. }
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ a6)
    as [unknown a7].
  cbn [snd] in Hu.
  unfold add_code_dep. cbn [acc_facts].
  apply nsa_add_facts; [ reflexivity | exact Hu ].
Qed.

Lemma nsa_prepare_binding (binding : static_binding) (a : acc) :
  nsa a -> nsa (prepare_binding binding a).
Proof.
  intros Ha. destruct binding; cbn [prepare_binding];
    [ apply nsa_prepare_code, Ha | exact Ha | exact Ha ].
Qed.

Lemma nsa_make_known (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  nsa a -> nsa (snd (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  intros Ha. unfold make_known_arity_apply_widget.
  destruct (fresh a) as [w a1] eqn:E1.
  destruct (fresh a1) as [called a2] eqn:E2.
  destruct (fresh a2) as [apply_node a3] eqn:E3.
  unfold fresh in E1, E2, E3.
  injection E1 as _ <-. injection E2 as _ <-. injection E3 as _ <-.
  cbn [snd]. apply nsa_cond_alias. apply nsa_add_facts; [ nsr_tac | ].
  rewrite note_simples_facts_eq. exact Ha.
Qed.

Lemma nsa_unknown_arity_apply_deps (returns : list node) (exn called : node) :
  ∀ (aws : list (list simple * node)) (a : acc),
    nsa a -> nsa (unknown_arity_apply_deps returns exn called aws a).
Proof.
  induction aws as [ | [first witness] rest IH]; intros a Ha;
    [ exact Ha | ].
  cbn [unknown_arity_apply_deps].
  assert (Ha1 : nsa (add_facts
                       (map (fun '(i, s) =>
                               Fact_argument (node_of_simple s) i witness)
                          (indexed first)
                        ++ [Fact_accessor exn (Field_return_of_call Exn)
                              witness;
                            Fact_accessor called
                              Field_code_id_of_call_witness witness]) a))
    by (apply nsa_add_facts; [ nsr_tac | exact Ha ]).
  destruct rest as [ | [first' next] rest'].
  - apply nsa_add_facts; [ nsr_tac | exact Ha1 ].
  - destruct (fresh _) as [v a2] eqn:Hf. apply IH.
    apply nsa_add_facts; [ reflexivity | ].
    unfold fresh in Hf. injection Hf as _ <-. exact Ha1.
Qed.

Lemma nsa_make_unknown (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  nsa a -> nsa (snd (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  intros Ha. unfold make_unknown_arity_apply_widget.
  destruct (fresh a) as [called a1] eqn:E1.
  unfold fresh in E1. injection E1 as _ <-.
  set (groups := at_least_one_group _).
  set (a2 := add_fact _ _).
  assert (Ha2 : nsa a2) by (apply nsa_add_facts; [ reflexivity | exact Ha ]).
  clearbody a2.
  pose proof (fresh_list_facts_eq (length groups) a2) as Hf.
  destruct (fresh_list _ a2) as [ws a3]. cbn [snd] in Hf.
  assert (Ha4 : nsa (unknown_arity_apply_deps returns exn called
                       (combine groups ws) (note_simples (ap_args ap) a3))).
  { apply nsa_unknown_arity_apply_deps.
    rewrite note_simples_facts_eq, Hf. exact Ha2. }
  destruct (fresh (unknown_arity_apply_deps _ _ _ _ _)) as [an a5] eqn:E5.
  unfold fresh in E5. injection E5 as _ <-.
  destruct ws as [ | first ws']; cbn [snd];
    [ exact Ha4 | apply nsa_cond_alias, Ha4 ].
Qed.

Lemma nsa_add_apply (env : traverse_env) (id : code_id)
    (callee call_widget : node) (b : bool) (a : acc) :
  nsa a -> nsa (add_apply env id callee call_widget b a).
Proof.
  intros Ha. unfold add_apply. destruct b; [ | exact Ha ].
  destruct (fresh a) as [c a1] eqn:E1.
  destruct (fresh a1) as [w a2] eqn:E2.
  unfold fresh in E1, E2. injection E1 as _ <-. injection E2 as _ <-.
  apply nsa_add_facts; [ reflexivity | exact Ha ].
Qed.

Lemma nsa_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  nsa a -> nsa (traverse_apply env ap a).
Proof.
  intros Ha. unfold traverse_apply.
  pose proof (nsa_find_cont env (ap_continuation ap)
                (length (unarize (ap_return_arity ap))) a Ha) as H1.
  destruct (find_cont _ _ _ a) as [return_args a1]. cbn [snd] in H1.
  pose proof (nsa_find_cont env (ap_exn_continuation ap)
                (S (length (ap_exn_extra_args ap))) a1 H1) as H2.
  destruct (find_cont _ _ _ a1) as [exn_params a2]. cbn [snd] in H2.
  assert (H3 : ∀ exn_arg a3,
             nsa a3 -> nsa (traverse_call_kind env ap exn_arg return_args a3)).
  { intros exn_arg a3 Ha3. unfold traverse_call_kind.
    destruct (ap_call_kind ap) as [id | | code_ids];
      [ | | pose proof (nsa_make_known env ap return_args exn_arg
                          (note_simple (ap_callee ap) a3)) as Hk;
            rewrite note_simple_facts_eq in Hk; specialize (Hk Ha3);
            destruct (make_known_arity_apply_widget _ _ _ _ _) as [cw a4];
            cbn [snd] in Hk;
            apply nsa_add_facts; [ reflexivity | exact Hk ] ].
    - pose proof (nsa_make_known env ap return_args exn_arg
                    (note_simple (ap_callee ap) a3)) as Hk.
      rewrite note_simple_facts_eq in Hk. specialize (Hk Ha3).
      destruct (make_known_arity_apply_widget _ _ _ _ _) as [cw a4].
      cbn [snd] in Hk.
      assert (Ha5 : nsa (add_fact (Fact_accessor cw
                                     (Field_call_witness
                                        Known_arity_code_pointer)
                                     (node_of_simple (ap_callee ap))) a4))
        by (apply nsa_add_facts; [ reflexivity | exact Hk ]).
      destruct (env_should_preserve_direct_calls env);
        try (apply nsa_add_apply, Ha5).
      destruct (negb _); [ apply nsa_add_apply, Ha5 | exact Ha5 ].
    - pose proof (nsa_make_unknown env ap return_args exn_arg
                    (note_simple (ap_callee ap) a3)) as Hk.
      rewrite note_simple_facts_eq in Hk. specialize (Hk Ha3).
      destruct (make_unknown_arity_apply_widget _ _ _ _ _) as [cw a4].
      cbn [snd] in Hk.
      apply nsa_add_facts; [ reflexivity | exact Hk ]. }
  destruct exn_params as [ | p extra].
  - destruct (fresh a2) as [e a3] eqn:E. unfold fresh in E.
    injection E as _ <-. apply H3. exact H2.
  - apply H3. apply nsa_add_facts; [ nsr_tac | ].
    rewrite note_simples_facts_eq. exact H2.
Qed.

Lemma nsa_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  nsa a -> nsa (apply_cont_deps env ac a).
Proof.
  intros Ha. unfold apply_cont_deps.
  pose proof (nsa_find_cont env (ac_continuation ac) (length (ac_args ac)) a
                Ha) as Hf.
  destruct (find_cont _ _ _ a) as [params a']. cbn [snd] in Hf.
  apply nsa_add_facts; [ nsr_tac | ].
  rewrite note_simples_facts_eq. exact Hf.
Qed.

Lemma nsa_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  nsa a -> nsa (traverse_switch env scrutinee arms a).
Proof.
  intros Ha. unfold traverse_switch.
  pose proof (nsa_cond_any_usage env scrutinee a Ha) as H.
  revert H. generalize (cond_any_usage env scrutinee a).
  induction (Keyed.compact Z.eqb arms) as [ | [d ac] l IH]; intros a' Ha';
    [ exact Ha' | ].
  simpl. apply IH. apply nsa_apply_cont_deps, Ha'.
Qed.

Lemma nsa_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  nsa a -> nsa (exn_handler_deps env h a).
Proof.
  intros Ha. unfold exn_handler_deps.
  destruct h as [[ | p params] metadata handler]; [ exact Ha | ].
  destruct (hm_is_exn_handler metadata); [ | exact Ha ].
  apply nsa_cond_any_usage, nsa_add_facts; [ reflexivity | exact Ha ].
Qed.

Lemma nsa_fold {B : Type} (step : acc -> B -> acc) (l : list B) :
  (∀ (b : B) (a : acc), b ∈ l -> nsa a -> nsa (step a b)) ->
  ∀ a : acc, nsa a -> nsa (List.fold_left step l a).
Proof.
  induction l as [ | b l IH]; intros Hstep a Ha; [ exact Ha | ].
  simpl. apply IH; [ intros b' a' Hb; apply Hstep; right; exact Hb | ].
  apply Hstep; [ left; reflexivity | exact Ha ].
Qed.

(* The traversal of a unit without projection reads no slot. *)
Lemma nsa_traverse :
  (∀ e : expr, projection_free e = true ->
     ∀ (env : traverse_env) (a : acc), nsa a -> nsa (traverse env e a))
  ∧ (∀ h : continuation_handler, handler_projection_free h = true ->
       ∀ (env : traverse_env) (a : acc), nsa a ->
         nsa (traverse_handler env h a)).
Proof.
  apply expr_handler_ind.
  - intros x kind d body IH Hpf env a Ha. simpl in Hpf.
    apply andb_true_iff in Hpf as [Hd Hbody]. cbn [traverse].
    apply (IH Hbody). apply nsa_traverse_named; [ exact Hd | exact Ha ].
  - intros bindings body IHc IH Hpf env a Ha. simpl in Hpf.
    apply andb_true_iff in Hpf as [Hbs Hbody]. cbn [traverse]. cbv zeta.
    apply (IH Hbody). rewrite forallb_forall in Hbs.
    apply nsa_fold; [ | apply nsa_fold; [ | apply nsa_fold ] ].
    + intros [id c | id | s const] a' Hin Ha'; cbn [traverse_binding];
        try exact Ha'.
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ]; [ | exact Ha' ].
      pose proof (Hbs _ Hin) as Hc. cbn in Hc.
      apply (IHc id (Code params mc ret exn md cbody) Hin Hc).
      apply nsa_add_facts; [ nsr_tac | exact Ha' ].
    + intros b a' _ Ha'. apply nsa_traverse_static_binding, Ha'.
    + intros b a' _ Ha'. apply nsa_prepare_binding, Ha'.
    + exact Ha.
  - intros k h body IHh IH Hpf env a Ha. simpl in Hpf.
    apply andb_true_iff in Hpf as [Hh Hbody]. cbn [traverse].
    apply (IH Hbody). apply nsa_exn_handler_deps. exact (IHh Hh env a Ha).
  - intros handlers body IHh IH Hpf env a Ha. simpl in Hpf.
    apply andb_true_iff in Hpf as [Hhs Hbody]. cbn [traverse].
    apply (IH Hbody). rewrite forallb_forall in Hhs.
    apply nsa_fold; [ | exact Ha ].
    intros [k h] a' Hin Ha'. exact (IHh k h Hin (Hhs _ Hin) _ a' Ha').
  - intros ap _ env a Ha. apply nsa_traverse_apply, Ha.
  - intros ac _ env a Ha. apply nsa_apply_cont_deps, Ha.
  - intros scrutinee arms _ env a Ha. apply nsa_traverse_switch, Ha.
  - intros _ env a Ha. exact Ha.
  - intros params metadata handler IH Hpf env a Ha. exact (IH Hpf env a Ha).
Qed.

Lemma nsa_add_closure_dep (cds : finite_map code_dep) (d : closure_dep)
    (a : acc) :
  nsa a -> nsa (add_closure_dep cds d a).
Proof.
  intros Ha. unfold add_closure_dep.
  destruct (lookup _ cds) as [cd | ].
  - apply nsa_add_facts; [ | exact Ha ].
    destruct (cd_unknown_arity_call_witnesses cd); reflexivity.
  - destruct (fresh a) as [w a1] eqn:E. unfold fresh in E.
    injection E as _ <-. apply nsa_add_facts; [ reflexivity | exact Ha ].
Qed.

Lemma nsa_resolve_delayed_deps (symbols : list symbol) (a : acc) :
  nsa a -> nsa (resolve_delayed_deps symbols a).
Proof.
  intros Ha. unfold resolve_delayed_deps.
  apply nsa_add_facts; [ nsr_tac | ].
  apply nsa_fold; [ intros d a' _; apply nsa_add_closure_dep | ].
  apply nsa_add_facts; [ | exact Ha ].
  induction (acc_apply_deps a) as [ | d l IH]; [ reflexivity | ].
  cbn [flat_map]. apply nsr_app; [ | exact IH ].
  unfold apply_dep_facts.
  destruct (lookup _ _); [ unfold alias_for_caller | ];
    destruct (ad_function_containing_apply_expr d); reflexivity.
Qed.

(* The facts of a unit without projection read no slot. *)
Lemma traverse_unit_reads_no_slot (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  projection_free e = true ->
  ∀ (to_ : node) (f : field) (base : node),
    (to_, f, base) ∈ f_accessor (traverse_unit flag u e) ->
    (∀ w : value_slot, f <> Field_value_slot w)
    ∧ (∀ fs : function_slot, f <> Field_function_slot fs).
Proof.
  intros Hpf to_ f base Hin.
  assert (Hall : nsa (resolve_delayed_deps (static_names symbols_of_binding e)
                        (add_fact (Fact_any_usage (Node_made 1))
                           (traverse (initial_env flag u e) e
                              initial_acc)))).
  { apply nsa_resolve_delayed_deps, nsa_add_facts; [ reflexivity | ].
    apply (proj1 nsa_traverse e Hpf). reflexivity. }
  rewrite traverse_unit_eq in Hin.
  cbn [facts_of_list f_accessor] in Hin.
  apply in_flat_map in Hin as (fct & Hfct & Hin).
  destruct fct; cbn in Hin; try contradiction.
  destruct Hin as [[= <- <- <-] | []].
  rewrite forallb_forall in Hall. specialize (Hall _ Hfct).
  split; intros ? ->; discriminate Hall.
Qed.

End NoSlotReads.

Lemma flow_sound_gen_local (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (n : nat) (G : graph) :
  graph_sub (graph_of_facts (traverse_unit flag u e)) G ->
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  closed is_local G (solution_of_tables T) ->
  local_reads_used is_local G (solution_of_tables T) ->
  ∃ located : location -> option node,
    match run n (State e [] [] empty_static_env empty_heap no_reads) with
    | Step s =>
      state_flows G (solution_of_tables T) (unit_kept e) (has_call_witness G)
        located s
    | Exit k vs hp _ =>
      exit_flows G (solution_of_tables T) (unit_kept e) (has_call_witness G)
        located u k vs
      ∧ heap_flows G (solution_of_tables T) (unit_kept e)
          (has_call_witness G) located hp
    | Undefined => True
    end.
Proof.
  intros Hsub Hlocal Hfrag Hc Hlrs.
  destruct (traverse_unit_facts_heap flag u e True Hfrag (or_introl I))
    as (noted & deps & cds & He0 & Hconsts0 & Himported & _ & Hdeps0 & Hnd
        & Hkeys).
  set (G0 := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  pose proof (static_names_edges G0 noted deps cds (unit_kept e)
                (unit_codes e) (static_names symbols_of_binding e)
                True True) as [Hnames _].
  destruct (Hnames e None (unit_conts u) He0) as [Hsym_names Hcode_names].
  pose proof (proj1 (expr_edges_mono G0 G Hsub noted deps cds (unit_kept e)
                       (unit_codes e) (static_names symbols_of_binding e)
                       True True)
                e None (unit_conts u) He0) as He.
  pose proof (sub_any_source G0 G Hsub _ Hconsts0) as Hconsts.
  assert (Hdeps : ∀ d : closure_dep, deps d -> closure_dep_facts G cds d).
  { intros d Hd. exact (closure_dep_facts_mono G0 G Hsub cds d (Hdeps0 d Hd)).
  }
  (* The code deps of the unit have the facts of their witnesses. *)
  assert (Hcds : ∀ (id : code_id) (cd : code_dep),
            lookup id cds = Some cd -> known_witness_edges G id cd).
  { intros id cd Hl.
    assert (Hid : id ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst (cds) (id, cd)).
      exact (lookup_in _ _ _ Hl). }
    destruct (Hcode_names id Hid) as (c & _ & Hcode).
    destruct c as [params mc ret exn metadata body].
    destruct Hcode as (cd' & Hin' & Hw & _).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hl. injection Hl as <-.
    exact (known_witness_edges_mono G0 G Hsub _ _ Hw). }
  (* The symbols of the unit are allocations of their own. *)
  assert (Hunit_syms : ∀ s : symbol,
            among s (static_names symbols_of_binding e) = true ->
            may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs. apply among_In in Hs.
    destruct (Hsym_names s Hs) as (const & Hb).
    destruct const as [ | slot syms decls vs | t fields | | | ];
      simpl in Hb; try contradiction.
    - destruct Hb as (Hslot & Hdep & _). unfold is_bound in Hslot.
      destruct (lookup slot decls) as [id | ] eqn:Hl; [ | discriminate Hslot ].
      exact (called_alloc is_local G sol cds Hc Hcds _ _
               (Hdeps _ (Hdep id eq_refl))).
    - destruct Hb as (_ & His & _).
      exact (alloc_self is_local G sol Hc _ _ _
               (sub_constructor G0 G Hsub _ _ _ His)
               (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
    - exact (boxed_self is_local G sol Hc Hconsts _ _
               (boxed_edges_mono G0 G Hsub _ _ Hb)).
    - exact (any_source_self is_local G sol Hc _
               (alias_consts_any_source is_local G sol Hc Hconsts _
                  (sub_alias G0 G Hsub _ _ Hb))).
    - destruct elements as [ | element elements ].
      + exact (any_source_self is_local G sol Hc _
                 (alias_consts_any_source is_local G sol Hc Hconsts _
                    (sub_alias G0 G Hsub _ _ Hb))).
      + destruct Hb as (_ & His & _).
        exact (alloc_self is_local G sol Hc _ _ _
                 (sub_constructor G0 G Hsub _ _ _ His)
                 (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))). }
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs.
    destruct (among s (static_names symbols_of_binding e)) eqn:Hunit;
      [ exact (Hunit_syms s Hunit) | ].
    apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _
             (sub_any_source G0 G Hsub _ (Himported s Hs Hunit))). }
  destruct (located_exists n (State e [] [] empty_static_env empty_heap
                                no_reads))
    as (located & Hloc).
  exists located.
  pose proof (WithLocal.run_flow_invariant is_local G sol noted deps cds
                (unit_kept e) (unit_codes e) (static_names symbols_of_binding e)
                u True located True Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps
                Hnd Hcds (fun _ => Hlrs) n _ Hloc
                (WithLocal.initial_flow_invariant G sol noted deps cds
                   (unit_kept e) (unit_codes e)
                   (static_names symbols_of_binding e) u True located True e
                   He)
                ltac:(apply initial_heap_ok))
    as Hrun.
  destruct (run n (State e [] [] empty_static_env empty_heap no_reads))
    as [s | k vs hp rd | ].
  - destruct Hrun as [Hinv [Hheap _]].
    destruct (WithLocal.flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hinv)
      as (Hvars & Hconts & Hst).
    apply (state_flows_weaken _ _ _ (called_by G cds));
      [ apply called_witness | ].
    exact (conj Hvars (conj Hconts (conj Hst Hheap))).
  - destruct Hrun as [Hexit Hheap]. split.
    + intros i v p Hv Hp.
      apply (flows_weaken _ _ _ (called_by G cds)); [ apply called_witness | ].
      exact (Hexit i v p Hv Hp).
    + exact (heap_flows_weaken _ _ _ _ _ _ _ (called_witness G cds) Hheap).
  - exact I.
Qed.

Theorem flow_sound_local : flow_sound_local_statement.
Proof.
  intros flag is_local u e T n Hlocal Hfrag Hcheck Hlrs.
  exact (flow_sound_gen_local flag is_local u e T n _ (graph_sub_refl _)
           Hlocal Hfrag (check_closed_sound is_local _ _ Hcheck)
           (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
              is_local _ _ Hlrs)).
Qed.

Theorem flow_sound_without_keep_alive_local :
  flow_sound_without_keep_alive_local_statement.
Proof.
  intros flag is_local u e T n Hlocal Hfrag Hcheck Hlrs.
  exact (flow_sound_gen_local flag is_local u e T n _
           (without_keep_alive_sub _) Hlocal Hfrag
           (check_closed_sound is_local _ _ Hcheck)
           (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
              is_local _ _ Hlrs)).
Qed.

Theorem old_premise_local_reads_used :
  old_premise_local_reads_used_statement.
Proof.
  intros flag is_local u e T Hlocal Hslots.
  unfold check_local_reads_used. apply forallb_forall.
  intros [[to_ f] base] Hin.
  destruct (is_local f) eqn:Hl;
    [ | rewrite andb_false_r; reflexivity ].
  exfalso. destruct Hslots as [Hno | Hpf];
    [ rewrite Hno in Hl; discriminate Hl | ].
  destruct (traverse_unit_reads_no_slot flag u e Hpf to_ f base Hin)
    as [Hw Hfs].
  destruct (Hlocal f Hl) as [(w & ->) | (fs & ->)];
    [ exact (Hw w eq_refl) | exact (Hfs fs eq_refl) ].
Qed.

(* The graph without the two facts of tupled code has fewer reads. *)
Lemma check_local_reads_used_without_keep_alive (is_local : field -> bool)
    (F : facts) (T : tables) :
  check_local_reads_used is_local F T = true ->
  check_local_reads_used is_local
    (FlambdaReaperTupledSpec.without_keep_alive F) T
  = true.
Proof.
  unfold check_local_reads_used. rewrite !forallb_forall.
  intros H x Hx. apply filter_In in Hx as [Hx _]. exact (H x Hx).
Qed.

Theorem flow_sound : flow_sound_statement.
Proof.
  intros flag is_local u e T n Hlocal Hslots Hfrag Hcheck.
  exact (flow_sound_local flag is_local u e T n Hlocal Hfrag Hcheck
           (old_premise_local_reads_used flag is_local u e T Hlocal Hslots)).
Qed.

Theorem flow_sound_without_keep_alive :
  flow_sound_without_keep_alive_statement.
Proof.
  intros flag is_local u e T n Hlocal Hslots Hfrag Hcheck.
  exact (flow_sound_without_keep_alive_local flag is_local u e T n Hlocal
           Hfrag Hcheck
           (check_local_reads_used_without_keep_alive is_local _ T
              (old_premise_local_reads_used flag is_local u e T Hlocal
                 Hslots))).
Qed.

Theorem unit_cont_nodes_used : unit_cont_nodes_used_statement.
Proof.
  intros flag u e k p Hfrag Hp.
  destruct (traverse_unit_facts_heap flag u e True Hfrag (or_introl I))
    as (_ & _ & _ & _ & _ & _ & Hused & _).
  exact (Hused k p Hp).
Qed.

(* The fragment with the primitives of the heap has every term of the
   fragment without them. *)
Lemma flow_fragment_heap :
  (∀ (e : expr) (codes : list (code_id * code)) (symbols : list symbol),
     flow_fragment false codes symbols e = true ->
     flow_fragment true codes symbols e = true)
  ∧ (∀ (h : continuation_handler) (codes : list (code_id * code))
       (symbols : list symbol),
       handler_flow_fragment false codes symbols h = true ->
       handler_flow_fragment true codes symbols h = true).
Proof.
  apply expr_handler_ind; simpl.
  - intros x kind d body IH codes symbols H.
    apply andb_true_iff in H as [Hd H]. rewrite (IH _ _ H), andb_true_r.
    destruct d; simpl in Hd |- *; first [ reflexivity | exact Hd ].
  - intros bindings body IHcode IH codes symbols H.
    apply andb_true_iff in H as [Hb H]. rewrite (IH _ _ H), andb_true_r.
    rewrite forallb_forall in Hb |- *. intros b Hin. specialize (Hb b Hin).
    destruct b as [id c | id | sym const]; simpl in Hb |- *;
      [ | reflexivity | exact Hb ].
    destruct c as [params my_closure ret exn metadata body']. simpl in *.
    apply andb_true_iff in Hb as [Hb Hbody'].
    pose proof (IHcode id _ Hin _ _ Hbody') as Hb'. simpl in Hb'.
    rewrite Hb', Hb. reflexivity.
  - intros k h body IHh IH codes symbols H.
    apply andb_true_iff in H as [Hh H].
    rewrite (IHh _ _ Hh), (IH _ _ H). reflexivity.
  - intros handlers body IHh IH codes symbols H.
    apply andb_true_iff in H as [Hh H]. rewrite (IH _ _ H), andb_true_r.
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    exact (IHh k h Hin _ _ (Hh (k, h) Hin)).
  - auto.
  - auto.
  - auto.
  - auto.
  - auto.
Qed.

Theorem in_flow_fragment_in_heap_fragment :
  in_flow_fragment_in_heap_fragment_statement.
Proof.
  intros e H. unfold in_flow_fragment, in_heap_fragment in *.
  apply andb_true_iff in H as [Hf Hd].
  rewrite (proj1 flow_fragment_heap e _ _ Hf), Hd. reflexivity.
Qed.


(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* Every tables of a list fails the check. *)
Lemma all_fail (is_local : field -> bool) (F : facts) (l : list tables) :
  forallb (fun T => negb (check_closed is_local F T)) l = true ->
  ∀ T : tables, T ∈ l -> check_closed is_local F T = false.
Proof.
  intros H T Hin. rewrite forallb_forall in H.
  apply negb_true_iff. exact (H T Hin).
Qed.

Theorem example : example_statement.
Proof.
  split; [ intros f Hf; discriminate Hf | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply all_fail; vm_compute; reflexivity | ].
  vm_compute. reflexivity.
Qed.

Theorem call_example : call_example_statement.
Proof.
  split; [ intros f Hf; discriminate Hf | ].
  split; [ reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply all_fail; vm_compute; reflexivity | ].
  vm_compute. reflexivity.
Qed.

Theorem indirect_example : indirect_example_statement.
Proof.
  split; [ intros f Hf; discriminate Hf | ].
  split; [ reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply all_fail; vm_compute; reflexivity | ].
  vm_compute. reflexivity.
Qed.

Theorem local_slot_counterexample : local_slot_counterexample_statement.
Proof.
  split.
  { intros f Hf. left. exists 60%positive.
    exact (field_eqb_eq _ _ Hf). }
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply all_fail; vm_compute; reflexivity | ].
  split; vm_compute; reflexivity.
Qed.

Theorem tupled_reverted_analysis_sound :
  tupled_reverted_analysis_sound_statement.
Proof.
  assert (Hfrag : in_heap_fragment FlambdaReaperTupledSpec.tupled_unit = true)
    by (vm_compute; reflexivity).
  split; [ exact Hfrag | ]. cbv zeta. intros n.
  destruct (FlambdaReaperTupledProofs.without_keep_alive_solution_passes
              FlambdaReaperTupledSpec.no_local_fields
              (or_introl eq_refl)) as [Hcheck _].
  exact (flow_sound_without_keep_alive Auto
           FlambdaReaperTupledSpec.no_local_fields
           FlambdaReaperTupledSpec.interface FlambdaReaperTupledSpec.tupled_unit
           FlambdaReaperTupledSpec.without_keep_alive_solution n
           ltac:(intros f H; discriminate H) (or_introl (fun f => eq_refl))
           Hfrag
           Hcheck).
Qed.

Theorem heap_example : heap_example_statement.
Proof.
  split; [ intros f Hf; discriminate Hf | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply all_fail; vm_compute; reflexivity | ].
  vm_compute. reflexivity.
Qed.
