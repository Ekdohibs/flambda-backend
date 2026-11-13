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
     the first has any usage ([flows_escape]), and at the node a field
     of it is read into ([flows_read]).
   - What the traversal makes: the facts of a term of the fragment, under
     the nodes of the parameters of the continuations in scope
     ([expr_edges]), are in the graph of the unit ([traverse_makes],
     [traverse_unit_facts]). The traversal threads an accumulator; a
     step of it keeps the facts it is given, so what it makes is in every
     later accumulator ([makes]).
   - An invariant of the run ([flow_invariant]): the facts of the
     expression to run are in the graph; the continuations of the state
     are handlers, each aligned with the nodes of its parameters and
     traversed with the continuations of its definition in scope
     ([conts_ok]); and the values are where [flows] says. Each step
     keeps it ([step_flow_invariant]). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the rules carry                                                     *)
(* ======================================================================== *)

Section Carry.

Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Hypothesis Hc : closed is_local G sol.

Abbreviation as_ n := (s_any_source sol n = true).
Abbreviation au n := (s_any_usage sol n = true).
Abbreviation hs n := (s_has_source sol n = true).
Abbreviation hu n := (s_has_usage sol n = true).

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
  s_alias sol to_ from = true -> flows G sol v from -> flows G sol v to_.
Proof.
  intros Hal. destruct v; simpl; not_in_fragment.
  - intros Has. exact (any_source_alias _ _ _ Hc _ _ Hal Has).
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
Qed.

(* A value at [m] is at a node with any source that has a usage only if
   [m] has any usage. *)
Lemma flows_escape (v : value) (to_ m : node) :
  as_ to_ -> (hu to_ -> au m) -> flows G sol v m -> flows G sol v to_.
Proof.
  intros Has Hm. destruct v; simpl; not_in_fragment.
  - intros _. exact Has.
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
Qed.

(* A node at which a value may be has a source. *)
Lemma flows_hs (v : value) (n : node) : flows G sol v n -> hs n.
Proof.
  destruct v; simpl; not_in_fragment.
  - apply as_hs.
  - intros (Hhs & _). exact Hhs.
  - intros (a & (Hhs & _) & _). exact Hhs.
  - intros (Has & _). exact (as_hs _ Has).
  - intros (a & bn & from & (Hhs & _) & _). exact Hhs.
Qed.

(* The read of the field [f] of a value that may come from [a]: what
   [a] holds there may be where it is read into, which is an alias of
   it, or a node with any source that has a usage only if what [a]
   holds has any usage. *)
Lemma flows_read (a base to_ from : node) (f : field) (v : value) :
  is_local f = false ->
  g_constructor G a f from ->
  g_accessor G to_ f base ->
  may_come_from sol a base ->
  flows G sol v from -> flows G sol v to_.
Proof.
  intros Hlocal Hcons Hacc (_ & Hsrc & _ & Hau & Hus) Hv.
  destruct (s_any_source sol base) eqn:Eb.
  - pose proof (any_source_accessor _ _ _ Hc _ _ _ Eb Hacc Hlocal) as Has.
    destruct (s_any_usage sol a) eqn:Ea.
    + apply (flows_escape v to_ from Has); [ | exact Hv ].
      intros _. exact (any_usage_constructor _ _ _ Hc _ _ _ Ea Hcons Hlocal).
    + destruct (s_has_usage sol to_) eqn:Ehu.
      * pose proof (has_usage_accessor _ _ _ Hc _ _ _ Ehu Hacc) as Hhu.
        assert (Eau : s_any_usage sol base = false).
        { destruct (s_any_usage sol base);
          [ discriminate (Hau eq_refl) | reflexivity ]. }
        pose proof (usages_accessor _ _ _ Hc _ _ _ Hacc Eau) as Hbb.
        destruct (Hus Hhu base Hbb) as [Hab | Hab];
          [ | congruence ].
        apply (flows_alias v to_ from); [ | exact Hv ].
        exact (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hab Ea Hacc).
      * apply (flows_escape v to_ from Has); [ | exact Hv ].
        intros Hhu. congruence.
  - destruct Hsrc as [Hsrc | Hsrc]; [ discriminate | ].
    apply (flows_alias v to_ from); [ | exact Hv ].
    exact (alias_accessor_source _ _ _ Hc _ _ _ _ _ Hacc Hsrc Eb Hcons).
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
    destruct (Hus Hhub base Hbb) as [Hab | Hab];
      [ | congruence ].
    exact (field_used_usage _ _ _ Hc _ _ _ _ _ Hcons Hab Hacc Hhu).
Qed.

End Carry.

Theorem flows_has_source : flows_has_source_statement.
Proof.
  intros is_local G sol v n Hc. exact (flows_hs is_local G sol Hc v n).
Qed.

Theorem read_field_used : read_field_used_statement.
Proof.
  intros is_local G sol a base to_ from f Hc.
  exact (field_used_read is_local G sol Hc a base to_ from f).
Qed.

Theorem read_flows : read_flows_statement.
Proof.
  intros is_local G sol a base to_ from f v Hc.
  exact (flows_read is_local G sol Hc a base to_ from f v).
Qed.

(* ======================================================================== *)
(* The facts of a term                                                      *)
(* ======================================================================== *)

(* What the traversal puts into the graph for a term of the fragment,
   under [ce], the nodes of the parameters of the continuations in
   scope; [noted] are the symbols that went through [simple_to_node].
   Only the facts that the proofs read are here. *)

Section Edges.

Variable G : graph.
Variable noted : symbol -> Prop.

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

Definition prim_edges (x : variable) (p : prim) (args : list simple)
    : Prop :=
  if prim_classified p args
  then classified_prim_edges x p args
  else default_prim_edges x args.

Definition named_edges (x : variable) (defining_expr : named) : Prop :=
  match defining_expr with
  | Simple s =>
    g_alias G (Node_variable x) (node_of_simple s) ∧ simple_noted s
  | Prim p args => prim_edges x p args
  | _ => False
  end.

Definition binding_edges (binding : static_binding) : Prop :=
  match binding with
  | Bind_symbol s (Static_const_block _ fields) =>
    block_edges (Node_symbol s) fields
  | _ => False
  end.

Definition apply_cont_edges (ce : finite_map (list node)) (ac : apply_cont)
    : Prop :=
  (∀ params : list node,
     lookup (ac_continuation ac) ce = Some params ->
     ∀ (p : node) (s : simple),
       (p, s) ∈ combine params (ac_args ac) ->
       g_alias G p (node_of_simple s))
  ∧ simples_noted (ac_args ac).

(* The continuations of a recursive group in scope. *)
Definition rec_conts (ce : finite_map (list node))
    (handlers : list (continuation * continuation_handler))
    : finite_map (list node) :=
  map (fun '(k, h) => (k, handler_params h)) handlers ++ ce.

Fixpoint expr_edges (ce : finite_map (list node)) (e : expr) : Prop :=
  match e with
  | Let x _ defining_expr body =>
    named_edges x defining_expr ∧ expr_edges ce body
  | Let_static bindings body =>
    every binding_edges bindings ∧ expr_edges ce body
  | Let_cont_non_recursive k h body =>
    handler_edges ce h ∧ expr_edges ((k, handler_params h) :: ce) body
  | Let_cont_recursive handlers body =>
    every (fun '(_, h) => handler_edges (rec_conts ce handlers) h) handlers
    ∧ expr_edges (rec_conts ce handlers) body
  | Apply _ => False
  | Apply_cont ac => apply_cont_edges ce ac
  | Switch _ arms =>
    every (fun '(_, ac) => apply_cont_edges ce ac) (Keyed.compact Z.eqb arms)
  | Invalid => True
  end
with handler_edges (ce : finite_map (list node)) (h : continuation_handler)
    : Prop :=
  match h with
  | Handler _ _ handler => expr_edges ce handler
  end.

End Edges.

(* Induction on the terms of the fragment, with the handlers of a
   recursive group. Code is not gone into. *)
Lemma expr_handler_ind (P : expr -> Prop) (Q : continuation_handler -> Prop) :
  (∀ x kind defining_expr body, P body -> P (Let x kind defining_expr body)) ->
  (∀ bindings body, P body -> P (Let_static bindings body)) ->
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
      + apply Hstatic, IH.
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
(* The traversal makes these facts                                          *)
(* ======================================================================== *)

(* An accumulator that has the facts and the symbols of another. *)
Definition acc_le (a a' : acc) : Prop :=
  (∀ f : fact, f ∈ acc_facts a -> f ∈ acc_facts a')
  ∧ (∀ s : symbol, s ∈ acc_symbols a -> s ∈ acc_symbols a').

Lemma acc_le_refl (a : acc) : acc_le a a.
Proof. split; auto. Qed.

Lemma acc_le_trans (a a' a'' : acc) :
  acc_le a a' -> acc_le a' a'' -> acc_le a a''.
Proof. intros [H1 H2] [H3 H4]. split; auto. Qed.

(* The graph of the facts of an accumulator, and its symbols. *)
Definition acc_graph (a : acc) : graph :=
  graph_of_facts (facts_of_list (acc_facts a)).

Definition acc_noted (a : acc) (s : symbol) : Prop := s ∈ acc_symbols a.

(* A step of the traversal that keeps what it is given and makes what
   [P] asks of any later accumulator. *)
Definition makes (step : acc -> acc) (P : acc -> Prop) : Prop :=
  ∀ a : acc, acc_le a (step a) ∧ ∀ a', acc_le (step a) a' -> P a'.

Lemma makes_compose (step step' : acc -> acc) (P P' : acc -> Prop) :
  makes step P -> makes step' P' ->
  makes (fun a => step' (step a)) (fun a' => P a' ∧ P' a').
Proof.
  intros H H' a. destruct (H a) as [Hle HP].
  destruct (H' (step a)) as [Hle' HP']. split.
  - exact (acc_le_trans _ _ _ Hle Hle').
  - intros a' Hle''. split.
    + apply HP. exact (acc_le_trans _ _ _ Hle' Hle'').
    + apply HP'. exact Hle''.
Qed.

Lemma makes_weaken (step : acc -> acc) (P P' : acc -> Prop) :
  (∀ a', P a' -> P' a') -> makes step P -> makes step P'.
Proof.
  intros Himpl H a. destruct (H a) as [Hle HP].
  split; [ exact Hle | intros a' Hle'; apply Himpl, HP, Hle' ].
Qed.

Lemma makes_ext (step step' : acc -> acc) (P : acc -> Prop) :
  (∀ a, step a = step' a) -> makes step P -> makes step' P.
Proof. intros Heq H a. rewrite <- Heq. apply H. Qed.

Lemma makes_nothing (step : acc -> acc) :
  (∀ a, acc_le a (step a)) -> makes step (fun _ => True).
Proof. intros H a. split; [ apply H | auto ]. Qed.

Lemma fold_left_makes {A : Type} (f : acc -> A -> acc)
    (P : A -> acc -> Prop) (l : list A) :
  (∀ x : A, x ∈ l -> makes (fun a => f a x) (P x)) ->
  makes (fun a => List.fold_left f l a) (fun a' => ∀ x : A, x ∈ l -> P x a').
Proof.
  induction l as [ | x l IH]; intros H a; simpl.
  - split; [ apply acc_le_refl | intros a' _ y [] ].
  - destruct (H x (or_introl eq_refl) a) as [Hle HP].
    destruct (IH (fun y Hy => H y (or_intror Hy)) (f a x)) as [Hle' HP'].
    split; [ exact (acc_le_trans _ _ _ Hle Hle') | ].
    intros a' Hle'' y [<- | Hy].
    + apply HP. exact (acc_le_trans _ _ _ Hle' Hle'').
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
Lemma add_facts_le (l : list fact) (a : acc) : acc_le a (add_facts l a).
Proof. split; simpl; [ intros f H; apply in_or_app; right; exact H | auto ].
Qed.

Lemma add_facts_later (l : list fact) (a a' : acc) (f : fact) :
  acc_le (add_facts l a) a' -> f ∈ l -> f ∈ acc_facts a'.
Proof.
  intros [Hle _] Hin. apply Hle. simpl. apply in_or_app. left. exact Hin.
Qed.

Lemma fresh_le (a : acc) : acc_le a (snd (fresh a)).
Proof. split; simpl; auto. Qed.

Lemma fresh_list_le (n : nat) (a : acc) : acc_le a (snd (fresh_list n a)).
Proof.
  revert a. induction n as [ | n IH]; intros a; [ apply acc_le_refl | ].
  change (fresh_list (S n) a)
    with (let '(x, a) := fresh a in
          let '(xs, a) := fresh_list n a in
          (x :: xs, a)).
  pose proof (fresh_le a) as Hle. destruct (fresh a) as [x a1].
  specialize (IH a1). destruct (fresh_list n a1) as [xs a2]. simpl in *.
  exact (acc_le_trans _ _ _ Hle IH).
Qed.

Lemma note_simple_le (s : simple) (a : acc) : acc_le a (note_simple s a).
Proof. destruct s; simpl; split; simpl; auto. Qed.

Lemma note_simple_noted (s : simple) (a a' : acc) :
  acc_le (note_simple s a) a' -> simple_noted (acc_noted a') s.
Proof.
  intros [_ Hle]. destruct s as [x | sym | c]; simpl; auto.
  apply Hle. left. reflexivity.
Qed.

Lemma note_simples_le (l : list simple) (a : acc) :
  acc_le a (note_simples l a).
Proof.
  unfold note_simples. revert a. induction l as [ | s l IH]; intros a; simpl.
  - apply acc_le_refl.
  - exact (acc_le_trans _ _ _ (note_simple_le s a) (IH _)).
Qed.

Lemma note_simples_noted (l : list simple) (a a' : acc) :
  acc_le (note_simples l a) a' -> simples_noted (acc_noted a') l.
Proof.
  unfold note_simples. revert a. induction l as [ | s l IH]; intros a Hle.
  - intros s [].
  - intros s' [<- | Hin]; simpl in Hle.
    + apply (note_simple_noted s a).
      exact (acc_le_trans _ _ _ (note_simples_le l _) Hle).
    + exact (IH _ Hle s' Hin).
Qed.

Lemma cond_any_usage_le (env : traverse_env) (s : simple) (a : acc) :
  acc_le a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage.
  destruct (env_current_code_id env);
    exact (acc_le_trans _ _ _ (note_simple_le s a) (add_facts_le _ _)).
Qed.

Lemma find_cont_le (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  acc_le a (snd (find_cont env k n a)).
Proof.
  unfold find_cont.
  destruct (lookup k (env_conts env)); [ apply acc_le_refl | ].
  destruct (fresh_list n a) as [params a'] eqn:E. simpl.
  pose proof (fresh_list_le n a) as Hle. rewrite E in Hle.
  exact (acc_le_trans _ _ _ Hle (add_facts_le _ _)).
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

(* The facts of each construct. *)

Lemma block_deps_makes (base : node) (fields : list simple) :
  makes (block_deps base fields)
    (fun a' => block_edges (acc_graph a') (acc_noted a') base fields).
Proof.
  intros a. unfold block_deps. split.
  - exact (acc_le_trans _ _ _ (note_simples_le fields a) (add_facts_le _ _)).
  - intros a' Hle. split; [ | split; [ | split ] ].
    + intros i s Hnth. apply facts_constructor.
      apply (add_facts_later _ _ _ _ Hle). apply in_or_app. left.
      apply in_map_iff. exists (i, s).
      split; [ reflexivity | exact (indexed_nth fields i s Hnth) ].
    + apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
      apply in_or_app. right. left. reflexivity.
    + apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
      apply in_or_app. right. right. left. reflexivity.
    + apply (note_simples_noted fields a a').
      exact (acc_le_trans _ _ _ (add_facts_le _ _) Hle).
Qed.

Lemma accessor_makes (x : variable) (f : field) (s : simple) :
  makes
    (fun a => add_fact (Fact_accessor (Node_variable x) f (node_of_simple s))
                (note_simple s a))
    (fun a' => accessor_edges (acc_graph a') (acc_noted a') x f s).
Proof.
  intros a. split.
  - exact (acc_le_trans _ _ _ (note_simple_le s a) (add_facts_le _ _)).
  - intros a' Hle. split.
    + apply facts_accessor. apply (add_facts_later _ _ _ _ Hle). left.
      reflexivity.
    + apply (note_simple_noted s a a').
      exact (acc_le_trans _ _ _ (add_facts_le _ _) Hle).
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
  destruct p as [u | b | v | h];
    [ destruct u | destruct b | destruct v | destruct h ];
    repeat (match goal with b : bool |- _ => destruct b end);
    destruct args as [ | s1 [ | s2 args]]; intros H;
    try discriminate H; reflexivity.
Qed.

(* The facts of a primitive, by the cases of [traverse_prim]. *)
Lemma traverse_prim_makes (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) :
  makes (traverse_prim env x p args)
    (fun a' => prim_edges (acc_graph a') (acc_noted a') x p args).
Proof.
  unfold prim_edges. destruct (prim_classified p args) eqn:Hcl.
  - destruct p as [u | b | v | h].
    + destruct u as [kind mut fld | [ | ] | | | ak | | ok | ik op | src dst
                    | | bn | bn | | | r];
        destruct args as [ | s1 [ | s2 args]]; simpl in Hcl;
        try discriminate Hcl.
      * (* [Block_load] *)
        intros a. destruct (accessor_makes x (Field_block (Z.to_nat fld) Value)
                              s1 a) as [Hle Hedges].
        unfold traverse_prim.
        destruct mut;
          [ split;
            [ exact (acc_le_trans _ _ _ Hle (add_facts_le _ _))
            | intros a' Hle'; apply Hedges;
              exact (acc_le_trans _ _ _ (add_facts_le _ _) Hle') ]
          | exact (conj Hle Hedges) | exact (conj Hle Hedges) ].
      * exact (accessor_makes x Field_is_int s1).
      * exact (accessor_makes x Field_get_tag s1).
      * exact (accessor_makes x (Field_boxed_number bn) s1).
      * (* [Box_number] *)
        intros a. split.
        -- exact (acc_le_trans _ _ _ (note_simple_le s1 a) (add_facts_le _ _)).
        -- intros a' Hle. split.
           ++ apply facts_constructor. apply (add_facts_later _ _ _ _ Hle).
              left. reflexivity.
           ++ apply (note_simple_noted s1 a a').
              exact (acc_le_trans _ _ _ (add_facts_le _ _) Hle).
    + discriminate Hcl.
    + destruct v; [ exact (block_deps_makes (Node_variable x) args)
                  | discriminate Hcl ].
    + destruct h; try discriminate Hcl.
      exact (block_deps_makes (Node_variable x) args).
  - apply (makes_ext (fun a =>
             add_facts
               (Fact_any_source (Node_variable x)
                :: map (fun from => Fact_use (Node_variable x) from)
                     (free_names_of_simples args))
               (if prim_has_arbitrary_effects p
                then cond_any_usage env (Var x) a else a))).
    { intros a. symmetry. exact (traverse_prim_default env x p args a Hcl). }
    intros a. split.
    + refine (acc_le_trans _ _ _ _ (add_facts_le _ _)).
      destruct (prim_has_arbitrary_effects p);
        [ apply cond_any_usage_le | apply acc_le_refl ].
    + intros a' Hle. split.
      * apply facts_any_source. apply (add_facts_later _ _ _ _ Hle). left.
        reflexivity.
      * intros from Hfrom. apply facts_use. apply (add_facts_later _ _ _ _ Hle).
        right. apply in_map_iff. exists from.
        split; [ reflexivity | exact Hfrom ].
Qed.

Lemma traverse_named_makes (env : traverse_env) (x : variable)
    (defining_expr : named) :
  named_in_flow_fragment defining_expr = true ->
  makes (traverse_named env x defining_expr)
    (fun a' => named_edges (acc_graph a') (acc_noted a') x defining_expr).
Proof.
  destruct defining_expr as [s | p args | | | | | ]; simpl;
    intros Hfrag; try discriminate Hfrag.
  - intros a. split.
    + exact (acc_le_trans _ _ _ (note_simple_le s a) (add_facts_le _ _)).
    + intros a' Hle. split.
      * apply facts_alias. apply (add_facts_later _ _ _ _ Hle). left.
        reflexivity.
      * apply (note_simple_noted s a a').
        exact (acc_le_trans _ _ _ (add_facts_le _ _) Hle).
  - apply traverse_prim_makes.
Qed.

(* An [Apply_cont] of a continuation in scope aliases its parameters to
   the arguments. *)
Lemma apply_cont_deps_makes (env : traverse_env) (ac : apply_cont) :
  makes (apply_cont_deps env ac)
    (fun a' => apply_cont_edges (acc_graph a') (acc_noted a')
                 (env_conts env) ac).
Proof.
  intros a. unfold apply_cont_deps.
  destruct (lookup (ac_continuation ac) (env_conts env)) as [params | ]
    eqn:Hlookup.
  - rewrite (find_cont_bound _ _ _ _ _ Hlookup). split.
    + exact (acc_le_trans _ _ _ (note_simples_le _ a) (add_facts_le _ _)).
    + intros a' Hle. split.
      * intros params' Hparams p s Hin.
        rewrite Hlookup in Hparams. injection Hparams as <-.
        apply facts_alias. apply (add_facts_later _ _ _ _ Hle).
        apply in_map_iff. exists (p, s). split; [ reflexivity | exact Hin ].
      * apply (note_simples_noted _ a a').
        exact (acc_le_trans _ _ _ (add_facts_le _ _) Hle).
  - pose proof (find_cont_le env (ac_continuation ac)
                  (length (ac_args ac)) a) as Hle0.
    destruct (find_cont env (ac_continuation ac) (length (ac_args ac)) a)
      as [params a1].
    simpl in Hle0. split.
    + refine (acc_le_trans _ _ _ Hle0 _).
      exact (acc_le_trans _ _ _ (note_simples_le _ a1) (add_facts_le _ _)).
    + intros a' Hle. split.
      * intros params' Hparams. rewrite Hlookup in Hparams. discriminate.
      * apply (note_simples_noted _ a1 a').
        exact (acc_le_trans _ _ _ (add_facts_le _ _) Hle).
Qed.

Lemma traverse_switch_makes (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  makes (traverse_switch env scrutinee arms)
    (fun a' =>
       every
         (fun '(_, ac) =>
            apply_cont_edges (acc_graph a') (acc_noted a') (env_conts env) ac)
         (Keyed.compact Z.eqb arms)).
Proof.
  assert (Hfold :
    makes
      (fun a =>
         List.fold_left (fun a '(_, ac) => apply_cont_deps env ac a)
           (Keyed.compact Z.eqb arms) a)
      (fun a' =>
         ∀ arm, arm ∈ Keyed.compact Z.eqb arms ->
           (fun '(_, ac) =>
              apply_cont_edges (acc_graph a') (acc_noted a') (env_conts env)
                ac) arm)).
  { apply fold_left_makes. intros [d ac] _. apply apply_cont_deps_makes. }
  pose proof (makes_compose _ _ _ _
                (makes_nothing _ (cond_any_usage_le env scrutinee)) Hfold)
    as Hboth.
  apply (makes_weaken _ _ _
           (fun a' (H : True ∧ _) => proj2 (every_in _ _ _) (proj2 H)))
    in Hboth.
  exact Hboth.
Qed.

Lemma exn_handler_deps_le (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  acc_le a (exn_handler_deps env h a).
Proof.
  destruct h as [[ | p params] metadata handler]; simpl;
    [ apply acc_le_refl | ].
  destruct (hm_is_exn_handler metadata); [ | apply acc_le_refl ].
  exact (acc_le_trans _ _ _ (add_facts_le _ _) (cond_any_usage_le _ _ _)).
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

(* The bindings of the fragment bind no code: the traversal goes through
   them once, by [traverse_static_binding]. *)
Lemma traverse_let_static (env : traverse_env)
    (bindings : list static_binding) (body : expr) (a : acc) :
  forallb binding_in_flow_fragment bindings = true ->
  traverse env (Let_static bindings body) a
  = traverse env body
      (List.fold_left (fun a b => traverse_static_binding env b a) bindings a).
Proof.
  intros Hfrag. simpl.
  assert (Hprepare : ∀ a,
            List.fold_left (fun a b => prepare_binding b a) bindings a = a).
  { clear a.
    induction bindings as [ | b bindings IH]; intros a; [ reflexivity | ].
    simpl in Hfrag |- *. apply andb_true_iff in Hfrag as [Hb Hfrag].
    destruct b as [ | | s const]; try discriminate Hb.
    simpl. exact (IH Hfrag a). }
  assert (Hbinding : ∀ a,
            List.fold_left (fun a b => traverse_binding env b a) bindings a
            = a).
  { clear a Hprepare.
    induction bindings as [ | b bindings IH]; intros a; [ reflexivity | ].
    simpl in Hfrag |- *. apply andb_true_iff in Hfrag as [Hb Hfrag].
    destruct b as [ | | s const]; try discriminate Hb.
    simpl. exact (IH Hfrag a). }
  rewrite Hprepare, Hbinding. reflexivity.
Qed.

(* The facts of a term of the fragment, and of a handler. *)
Lemma traverse_makes :
  (∀ e : expr,
     in_flow_fragment e = true ->
     ∀ env : traverse_env,
       makes (traverse env e)
         (fun a' => expr_edges (acc_graph a') (acc_noted a') (env_conts env) e))
  ∧ (∀ h : continuation_handler,
       handler_in_flow_fragment h = true ->
       ∀ env : traverse_env,
         makes (traverse_handler env h)
           (fun a' =>
              handler_edges (acc_graph a') (acc_noted a') (env_conts env) h)).
Proof.
  apply expr_handler_ind.
  - (* [Let] *)
    intros x kind d body IH Hfrag env. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hd Hbody].
    exact (makes_compose _ _ _ _ (traverse_named_makes env x d Hd)
             (IH Hbody env)).
  - (* [Let_static] *)
    intros bindings body IH Hfrag env. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hbindings Hbody].
    apply (makes_ext
             (fun a =>
                traverse env body
                  (List.fold_left (fun a b => traverse_static_binding env b a)
                     bindings a))).
    { intros a. symmetry.
      exact (traverse_let_static env bindings body a Hbindings). }
    assert (Hfold :
      makes
        (fun a =>
           List.fold_left (fun a b => traverse_static_binding env b a)
             bindings a)
        (fun a' =>
           ∀ b, b ∈ bindings -> binding_edges (acc_graph a') (acc_noted a') b)).
    { apply fold_left_makes. intros b Hin. rewrite forallb_forall in Hbindings.
      specialize (Hbindings b Hin).
      destruct b as [ | | s [ | | t fields | | | ]]; try discriminate Hbindings.
      exact (block_deps_makes (Node_symbol s) fields). }
    pose proof (makes_compose _ _ _ _ Hfold (IH Hbody env)) as Hboth.
    eapply makes_weaken; [ | exact Hboth ].
    intros a' [Hall Hb]. split; [ | exact Hb ].
    apply (every_in _ _ _). exact Hall.
  - (* [Let_cont_non_recursive] *)
    intros k h body IHh IH Hfrag env. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hh Hbody].
    pose proof (makes_compose _ _ _ _
                  (makes_compose _ _ _ _ (IHh Hh env)
                     (makes_nothing _ (exn_handler_deps_le env h)))
                  (IH Hbody (add_cont env k (handler_params h)))) as Hboth.
    eapply makes_weaken; [ | exact Hboth ].
    intros a' [[Hh' _] Hb]. split; [ exact Hh' | exact Hb ].
  - (* [Let_cont_recursive] *)
    intros handlers body IHh IH Hfrag env. simpl in Hfrag.
    apply andb_true_iff in Hfrag as [Hhandlers Hbody].
    rewrite forallb_forall in Hhandlers.
    set (env' := add_rec_conts env handlers).
    apply (makes_ext
             (fun a =>
                traverse env' body
                  (List.fold_left (fun a '(_, h) => traverse_handler env' h a)
                     handlers a))); [ reflexivity | ].
    assert (Hfold :
      makes
        (fun a =>
           List.fold_left (fun a '(_, h) => traverse_handler env' h a)
             handlers a)
        (fun a' =>
           ∀ kh, kh ∈ handlers ->
             (fun '(_, h) =>
                handler_edges (acc_graph a') (acc_noted a') (env_conts env') h)
               kh)).
    { apply fold_left_makes. intros [k h] Hin.
      exact (IHh k h Hin (Hhandlers _ Hin) env'). }
    pose proof (makes_compose _ _ _ _ Hfold (IH Hbody env')) as Hboth.
    eapply makes_weaken; [ | exact Hboth ].
    intros a' [Hall Hb]. unfold env' in Hall, Hb.
    rewrite add_rec_conts_conts in Hall, Hb. split; [ | exact Hb ].
    apply (every_in _ _ _). exact Hall.
  - (* [Apply] *)
    intros ap Hfrag. discriminate Hfrag.
  - (* [Apply_cont] *)
    intros ac _ env. apply apply_cont_deps_makes.
  - (* [Switch] *)
    intros scrutinee arms _ env. apply traverse_switch_makes.
  - (* [Invalid] *)
    intros _ env a. split; [ apply acc_le_refl | intros; exact I ].
  - (* A handler *)
    intros params metadata handler IH Hfrag env. exact (IH Hfrag env).
Qed.

(* The delayed deps add to what they are given, and give any source to
   the symbols that went through [simple_to_node] and that the unit
   does not bind. *)
Lemma add_closure_dep_le (code_deps : finite_map code_dep) (d : closure_dep)
    (a : acc) :
  acc_le a (add_closure_dep code_deps d a)
  ∧ acc_symbols (add_closure_dep code_deps d a) = acc_symbols a.
Proof.
  unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) code_deps).
  - split; [ apply add_facts_le | reflexivity ].
  - pose proof (fresh_le a) as Hle. destruct (fresh a) as [w a1] eqn:E.
    simpl in Hle. split.
    + exact (acc_le_trans _ _ _ Hle (add_facts_le _ _)).
    + simpl. injection E as _ <-. reflexivity.
Qed.

Lemma fold_closure_deps_le (code_deps : finite_map code_dep)
    (l : list closure_dep) (a : acc) :
  acc_le a (List.fold_left (fun a d => add_closure_dep code_deps d a) l a)
  ∧ acc_symbols (List.fold_left (fun a d => add_closure_dep code_deps d a) l a)
    = acc_symbols a.
Proof.
  revert a. induction l as [ | d l IH]; intros a; simpl.
  - split; [ apply acc_le_refl | reflexivity ].
  - destruct (add_closure_dep_le code_deps d a) as [Hle Hsyms].
    destruct (IH (add_closure_dep code_deps d a)) as [Hle' Hsyms'].
    split; [ exact (acc_le_trans _ _ _ Hle Hle') | congruence ].
Qed.

Lemma resolve_delayed_deps_le (unit_symbols : list symbol) (a : acc) :
  acc_le a (resolve_delayed_deps unit_symbols a)
  ∧ ∀ s : symbol,
      s ∈ acc_symbols (resolve_delayed_deps unit_symbols a) ->
      among s unit_symbols = false ->
      Fact_any_source (Node_symbol s)
      ∈ acc_facts (resolve_delayed_deps unit_symbols a).
Proof.
  unfold resolve_delayed_deps.
  set (a1 := add_facts _ a).
  destruct (fold_closure_deps_le (acc_code_deps a) (acc_closure_deps a1) a1)
    as [Hle Hsyms].
  set (a2 := List.fold_left _ _ a1) in *.
  split.
  - refine (acc_le_trans _ _ _ (add_facts_le _ _) _).
    exact (acc_le_trans _ _ _ Hle (add_facts_le _ _)).
  - intros s Hin Hunit. simpl in Hin |- *. apply in_or_app. left.
    apply in_map_iff. exists s. split; [ reflexivity | ].
    apply among_In. rewrite among_without, among_dedup, Hunit.
    rewrite andb_true_r. apply among_In. exact Hin.
Qed.

(* The symbols that a term binds are allocations with the constructor
   facts of a block. *)
Lemma static_names_edges (G : graph) (noted : symbol -> Prop) :
  (∀ e : expr,
     ∀ ce : finite_map (list node),
       expr_edges G noted ce e ->
       ∀ s : symbol, s ∈ static_names symbols_of_binding e ->
       g_constructor G (Node_symbol s) Field_is_int Node_all_constants)
  ∧ (∀ h : continuation_handler,
       ∀ ce : finite_map (list node),
         handler_edges G noted ce h ->
         ∀ s : symbol, s ∈ handler_static_names symbols_of_binding h ->
         g_constructor G (Node_symbol s) Field_is_int Node_all_constants).
Proof.
  apply expr_handler_ind; simpl.
  - intros x kind d body IH ce [_ He]. exact (IH ce He).
  - intros bindings body IH ce [Hbindings He] s Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH ce He s Hin) ].
    apply in_flat_map in Hin as (b & Hb & Hs).
    pose proof (proj1 (every_in _ _ _) Hbindings b Hb) as Hbe.
    destruct b as [ | | s' [ | | t fields | | | ]]; simpl in Hs, Hbe;
      try contradiction.
    destruct Hs as [<- | []]. exact (proj1 (proj2 Hbe)).
  - intros k h body IHh IH ce [Hh He] s Hin.
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh ce Hh s Hin) | exact (IH _ He s Hin) ].
  - intros handlers body IHh IH ce [Hhandlers He] s Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH _ He s Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hs).
    pose proof (proj1 (every_in _ _ _) Hhandlers _ Hkh) as Hh.
    exact (IHh k h Hkh _ Hh s Hs).
  - intros ap _ [].
  - intros ac _ _ _ [].
  - intros scrutinee arms _ _ _ [].
  - intros _ _ _ [].
  - intros params metadata handler IH ce He. exact (IH ce He).
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
     env_unit_code_ids := static_names all_code_ids_of_binding e |}.

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
Lemma traverse_unit_facts (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  in_flow_fragment e = true ->
  ∃ noted : symbol -> Prop,
    expr_edges (graph_of_facts (traverse_unit flag u e)) noted (unit_conts u) e
    ∧ g_any_source (graph_of_facts (traverse_unit flag u e)) Node_all_constants
    ∧ (∀ s : symbol,
         noted s ->
         among s (static_names symbols_of_binding e) = false ->
         g_any_source (graph_of_facts (traverse_unit flag u e))
           (Node_symbol s))
    ∧ (∀ (k : continuation) (p : node),
         p ∈ unit_cont_nodes u k ->
         g_any_usage (graph_of_facts (traverse_unit flag u e)) p).
Proof.
  intros Hfrag. rewrite traverse_unit_eq.
  destruct (proj1 traverse_makes e Hfrag (initial_env flag u e) initial_acc)
    as [Hle4 Hedges].
  set (a4 := traverse (initial_env flag u e) e initial_acc) in *.
  set (a5 := add_fact (Fact_any_usage (Node_made 1)) a4).
  destruct (resolve_delayed_deps_le (static_names symbols_of_binding e) a5)
    as [Hle6 Hsyms].
  set (a6 := resolve_delayed_deps _ a5) in *.
  pose proof (acc_le_trans _ _ _ (add_facts_le _ a4) Hle6) as Hle46.
  exists (acc_noted a6). split; [ | split; [ | split ] ].
  - exact (Hedges a6 Hle46).
  - apply facts_any_source. apply Hle46, Hle4. right. left. reflexivity.
  - intros s Hs Hunit. apply facts_any_source. exact (Hsyms s Hs Hunit).
  - intros k p Hp. unfold unit_cont_nodes, unit_conts in Hp.
    simpl in Hp. apply facts_any_usage.
    destruct (Pos.eqb k (unit_exn u));
      [ | destruct (Pos.eqb k (unit_return u)) ];
      simpl in Hp; [ | | contradiction ]; destruct Hp as [<- | []].
    + apply Hle46, Hle4. left. reflexivity.
    + apply Hle6. left. reflexivity.
Qed.

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

Section Run.

Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Variable noted : symbol -> Prop.
Variable u : unit_interface.
Hypothesis Hc : closed is_local G sol.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hconsts : g_any_source G Node_all_constants.
Hypothesis Hsyms :
  ∀ s : symbol, noted s -> may_come_from sol (Node_symbol s) (Node_symbol s).

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

Lemma simple_flows (vars : env) (s : simple) (v : value) :
  env_flows G sol vars -> simple_noted noted s ->
  eval_simple vars s = Some v -> flows G sol v (node_of_simple s).
Proof.
  intros Hvars Hnoted Hs. destruct s as [x | sym | c]; simpl in Hs, Hnoted |- *.
  - exact (Hvars x v Hs).
  - injection Hs as <-. exact (Hsyms sym Hnoted).
  - injection Hs as <-. exact consts_any_source.
Qed.

(* What is said of a block at an allocation. *)
Definition block_structure (a : node) (fields : list value) : Prop :=
  g_constructor G a Field_is_int Node_all_constants
  ∧ g_constructor G a Field_get_tag Node_all_constants
  ∧ fields_from
      (fun i field =>
         ∃ from : node,
           g_constructor G a (Field_block i Value) from
           ∧ flows G sol field from)
      0 fields.

Lemma flows_block (tag : Z) (fields : list value) (n : node) :
  flows G sol (Value_block tag fields) n
  <-> ∃ a : node, may_come_from sol a n ∧ block_structure a fields.
Proof. split; intros H; exact H. Qed.

Lemma block_structure_made (a : node) (fields : list simple) (vars : env)
    (vs : list value) :
  block_edges G noted a fields -> env_flows G sol vars ->
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
  statics_flow G sol statics -> flows G sol v n ->
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
  exact (flows_read _ _ _ Hc _ _ _ _ _ (Value_const c) Hl Hcons Hacc Ha
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
  statics_flow G sol statics -> env_flows G sol vars ->
  default_prim_edges G x args ->
  prim_classified p args = false ->
  eval_simples vars args = Some vs ->
  eval_prim statics p vs = Some v ->
  flows G sol v (Node_variable x).
Proof.
  intros Hst Hvars [Has Huse] Hcl Hvs Hev.
  pose proof (any_source_input _ _ _ Hc _ Has) as Hasx.
  (* Each argument is a constant, or is used by [x]. *)
  assert (Harg : ∀ (s : simple) (w : value),
    s ∈ args -> eval_simple vars s = Some w ->
    (∃ c : const, w = Value_const c)
    ∨ (∃ y : variable,
         flows G sol w (Node_variable y)
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
    + exact (flows_escape _ _ _ Hc _ _ _ Hasx Hu Hy).
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
      apply (flows_escape _ _ _ Hc v _ m Hasx); [ | exact Hm ].
      intros Hhu. apply Hmu, (au_hu _ _ _ Hc), Hu, Hhu.
    + destruct (Hst sym _ Hl) as [_ []].
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
Lemma named_flows (statics : static_env) (vars : env) (x : variable)
    (d : named) (v : value) :
  statics_flow G sol statics -> env_flows G sol vars ->
  named_edges G noted x d ->
  eval_named statics vars d = Some v ->
  flows G sol v (Node_variable x).
Proof.
  intros Hst Hvars Hedges Hev.
  destruct d as [s | p args | | | | | ]; simpl in Hedges; try contradiction;
    simpl in Hev.
  - destruct Hedges as [Hal Hnoted].
    apply (flows_alias _ _ _ Hc v _ _ (alias_input _ _ _ Hc _ _ Hal)).
    exact (simple_flows _ _ _ Hvars Hnoted Hev).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    unfold prim_edges in Hedges.
    destruct (prim_classified p args) eqn:Hcl;
      [ | exact (default_flows _ _ _ _ _ _ _ Hst Hvars Hedges Hcl Hvs Hev) ].
    destruct p as [p' | b | p' | h].
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
        refine (flows_read _ _ _ Hc _ _ _ _ _ _ _ Hcons Hedge Ha Hfrom).
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
           refine (flows_read _ _ _ Hc _ _ _ _ _ (Value_const c) _ Hcons Hedge
                     Ha Hfrom).
           not_local_field.
        -- destruct (Hst s _ Hl) as [_ []].
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
Qed.

(* A [Let_static] of the fragment binds symbols to blocks with the facts
   of the graph. *)
Lemma statics_flow_bindings (vars : env) (bindings : list static_binding)
    (statics statics' : static_env) :
  env_flows G sol vars -> every (binding_edges G noted) bindings ->
  statics_flow G sol statics ->
  eval_static_bindings vars bindings statics = Some statics' ->
  statics_flow G sol statics'.
Proof.
  intros Hvars. revert statics.
  induction bindings as [ | b bindings IH];
    intros statics Hedges Hst Heval; simpl in Heval.
  - injection Heval as <-. exact Hst.
  - destruct Hedges as [Hb Hbindings].
    destruct b as [ | | s [ | | t fields | | | ]]; simpl in Hb;
      try contradiction.
    destruct (lookup s (static_values statics)); [ discriminate Heval | ].
    simpl in Heval.
    destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
      [ | discriminate Heval ].
    refine (IH _ Hbindings _ Heval).
    intros s' allocated Hl. simpl in Hl.
    destruct (Pos.eqb_spec s' s) as [-> | _]; [ | exact (Hst s' allocated Hl) ].
    injection Hl as <-. destruct Hb as (Hcons & His & Hgt & Hnoted). split.
    + exact (alloc_self _ _ _ His (as_hs _ _ _ Hc _ consts_any_source)).
    + exact (block_structure_made _ _ _ _
               (conj Hcons (conj His (conj Hgt Hnoted))) Hvars Hvs).
Qed.

(* ------------------------------------------------------------------------ *)
(* Continuations                                                            *)
(* ------------------------------------------------------------------------ *)

(* The continuations of a state and the nodes of their parameters, as the
   traversal had them where the current expression was traversed: each
   continuation bound is a handler whose environment is at its nodes,
   whose own continuations are so, and which was traversed, with the
   handlers of its group, with the continuations of its definition in
   scope. Below them, the continuations of the unit. *)
Inductive conts_ok : cont_env -> finite_map (list node) -> Prop :=
  | Conts_ok_nil : conts_ok [] (unit_conts u)
  | Conts_ok_cons (k : continuation) (h : continuation_handler) (vars : env)
      (conts : cont_env)
      (recursive : list (continuation * continuation_handler))
      (K : cont_env) (ce ce0 : finite_map (list node)) :
    env_flows G sol vars ->
    conts_ok conts ce0 ->
    (∀ (k' : continuation) (h' : continuation_handler),
       (k', h') ∈ recursive ->
       handler_edges G noted (rec_conts ce0 recursive) h') ->
    handler_edges G noted (rec_conts ce0 recursive) h ->
    conts_ok K ce ->
    conts_ok ((k, Closure h vars conts recursive) :: K)
      ((k, handler_params h) :: ce).

(* A continuation that the state does not bind has the nodes of the
   unit's, if any. *)
Lemma conts_ok_none (K : cont_env) (ce : finite_map (list node))
    (k : continuation) :
  conts_ok K ce -> lookup k K = None -> lookup k ce = lookup k (unit_conts u).
Proof.
  intros Hok. induction Hok as [ | k' h vars conts recursive K ce ce0
                                  _ _ _ _ _ _ IH]; [ reflexivity | ].
  simpl. destruct (Pos.eqb k k'); [ discriminate | exact IH ].
Qed.

(* What a continuation bound stands for, and the nodes of its
   parameters. *)
Lemma conts_ok_some (K : cont_env) (ce : finite_map (list node))
    (k : continuation) (c : cont_value) :
  conts_ok K ce -> lookup k K = Some c ->
  ∃ (h : continuation_handler) (vars : env) (conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (ce0 : finite_map (list node)),
    c = Closure h vars conts recursive
    ∧ lookup k ce = Some (handler_params h)
    ∧ env_flows G sol vars
    ∧ conts_ok conts ce0
    ∧ (∀ (k' : continuation) (h' : continuation_handler),
         (k', h') ∈ recursive ->
         handler_edges G noted (rec_conts ce0 recursive) h')
    ∧ handler_edges G noted (rec_conts ce0 recursive) h.
Proof.
  intros Hok. induction Hok as [ | k' h vars conts recursive K ce ce0
                                  Hvars Hconts _ Hrec Hh _ IH];
    [ discriminate | ].
  simpl. destruct (Pos.eqb k k'); [ | exact IH ].
  intros [= <-]. exists h, vars, conts, recursive, ce0. auto 7.
Qed.

Lemma conts_ok_flows (K : cont_env) (ce : finite_map (list node)) :
  conts_ok K ce ->
  ∀ (k : continuation) (c : cont_value), (k, c) ∈ K -> cont_flows G sol c.
Proof.
  intros Hok. induction Hok as [ | k' h vars conts recursive K ce ce0
                                  Hvars _ IHconts _ _ _ IH];
    intros k c Hin; [ destruct Hin | ].
  destruct Hin as [[= <- <-] | Hin]; [ | exact (IH k c Hin) ].
  apply Closure_flows; [ exact Hvars | exact IHconts ].
Qed.

(* The closures of a recursive group in front of the continuations of its
   definition. *)
Lemma conts_ok_rec (vars : env) (conts : cont_env)
    (ce0 : finite_map (list node))
    (recursive : list (continuation * continuation_handler)) :
  env_flows G sol vars -> conts_ok conts ce0 ->
  (∀ (k : continuation) (h : continuation_handler),
     (k, h) ∈ recursive ->
     handler_edges G noted (rec_conts ce0 recursive) h) ->
  conts_ok (rec_closures recursive vars conts ++ conts)
    (rec_conts ce0 recursive).
Proof.
  intros Hvars Hconts Hrec. unfold rec_closures, rec_conts.
  assert (Hgen : ∀ l : list (continuation * continuation_handler),
            (∀ kh, kh ∈ l -> kh ∈ recursive) ->
            conts_ok
              (map (fun '(k, h) => (k, Closure h vars conts recursive)) l
               ++ conts)
              (map (fun '(k, h) => (k, handler_params h)) l ++ ce0)).
  { induction l as [ | [k h] l IH]; intros Hl; [ exact Hconts | ].
    simpl. apply Conts_ok_cons with (ce0 := ce0).
    - exact Hvars.
    - exact Hconts.
    - exact Hrec.
    - exact (Hrec k h (Hl _ (or_introl eq_refl))).
    - apply IH. intros kh Hkh. apply Hl. right. exact Hkh. }
  exact (Hgen recursive (fun _ H => H)).
Qed.

(* ------------------------------------------------------------------------ *)
(* Steps                                                                    *)
(* ------------------------------------------------------------------------ *)

(* The invariant of the run: the facts of the expression are in the
   graph, with the nodes of the continuations in scope, and the values
   of the state are where the solution says. *)
Definition flow_invariant (s : state) : Prop :=
  ∃ ce : finite_map (list node),
    expr_edges G noted ce (state_expr s)
    ∧ env_flows G sol (state_vars s)
    ∧ conts_ok (state_conts s) ce
    ∧ statics_flow G sol (state_statics s).

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

(* The arguments of an [Apply_cont] are at the nodes of the parameters. *)
Lemma apply_cont_args_flow (vars : env) (ce : finite_map (list node))
    (ac : apply_cont) (vs : list value) (params : list node) :
  env_flows G sol vars -> apply_cont_edges G noted ce ac ->
  eval_simples vars (ac_args ac) = Some vs ->
  lookup (ac_continuation ac) ce = Some params ->
  ∀ (i : nat) (v : value) (p : node),
    nth_error vs i = Some v -> nth_error params i = Some p ->
    flows G sol v p.
Proof.
  intros Hvars [Haliases Hnoted] Hvs Hparams i v p Hv Hp.
  destruct (eval_simples_nth _ _ _ _ _ Hvs Hv) as (s & Hs & Hsv).
  apply (flows_alias _ _ _ Hc v _ (node_of_simple s)).
  - apply (alias_input _ _ _ Hc). apply (Haliases params Hparams p s).
    exact (nth_error_combine _ _ _ _ _ Hp Hs).
  - exact (simple_flows _ _ _ Hvars (Hnoted s (nth_error_In _ _ Hs)) Hsv).
Qed.

(* A step keeps the invariant, and an exit has its values at the nodes
   of the continuation of the unit. *)
Lemma step_flow_invariant (s : state) :
  flow_invariant s ->
  match step s with
  | Step s' => flow_invariant s'
  | Exit k vs _ => exit_flows G sol u k vs
  | Undefined => True
  end.
Proof.
  intros (ce & He & Hvars & Hconts & Hst). rewrite (state_eta s).
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl in He.
  - (* [Let] *)
    rewrite step_let.
    destruct (eval_named (state_statics s) (state_vars s) d) as [v | ] eqn:Hev;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    destruct He as [Hd Hbody]. exists ce. simpl_state.
    split; [ exact Hbody | split; [ | split; [ exact Hconts | exact Hst ] ] ].
    intros y w Hy. simpl in Hy. destruct (Pos.eqb_spec y x) as [-> | _].
    + injection Hy as <-. exact (named_flows _ _ _ _ _ Hst Hvars Hd Hev).
    + exact (Hvars y w Hy).
  - (* [Let_static] *)
    rewrite step_let_static.
    destruct (eval_static_bindings (state_vars s) bindings (state_statics s))
      as [statics' | ] eqn:Hb; [ | exact I ].
    destruct He as [Hbindings Hbody]. exists ce. simpl_state.
    split; [ exact Hbody | split; [ exact Hvars | split; [ exact Hconts | ] ] ].
    exact (statics_flow_bindings _ _ _ _ Hvars Hbindings Hst Hb).
  - (* [Let_cont_non_recursive] *)
    rewrite step_let_cont_non_recursive.
    destruct He as [Hh Hbody]. exists ((k, handler_params h) :: ce).
    simpl_state.
    split; [ exact Hbody | split; [ exact Hvars | split; [ | exact Hst ] ] ].
    apply Conts_ok_cons with (ce0 := ce); try assumption.
    intros k' h' [].
  - (* [Let_cont_recursive] *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    destruct He as [Hhandlers Hbody]. exists (rec_conts ce handlers).
    simpl_state.
    split; [ exact Hbody | split; [ exact Hvars | split; [ | exact Hst ] ] ].
    apply conts_ok_rec; [ exact Hvars | exact Hconts | ].
    intros k h Hin. exact (proj1 (every_in _ _ _) Hhandlers (k, h) Hin).
  - (* [Apply] *)
    contradiction.
  - (* [Apply_cont] *)
    rewrite step_apply_cont.
    destruct (eval_simples (state_vars s) (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation, resolve.
    destruct (lookup (ac_continuation ac) (state_conts s)) as [c | ] eqn:Hl.
    + destruct (conts_ok_some _ _ _ _ Hconts Hl)
        as (h & vars & conts & recursive & ce0 & -> & Hparams & Hvars'
            & Hconts' & Hrec & Hh).
      destruct h as [params metadata handler]. cbn [apply_cont_value].
      destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
        [ | exact I ].
      destruct (bind_params params vs vars) as [handler_vars | ] eqn:Hbind;
        [ | exact I ].
      exists (rec_conts ce0 recursive). simpl_state.
      split; [ exact Hh | split; [ | split; [ | exact Hst ] ] ].
      * intros y w Hy.
        destruct (bind_params_lookup _ _ _ _ _ _ Hbind Hy)
          as [Hold | (i & Hp & Hw)]; [ exact (Hvars' y w Hold) | ].
        refine (apply_cont_args_flow _ _ _ _ _ Hvars He Hvs Hparams i w _ Hw _).
        simpl. apply map_nth_error. exact Hp.
      * exact (conts_ok_rec _ _ _ _ Hvars' Hconts' Hrec).
    + simpl. intros i v p Hv Hp. unfold unit_cont_nodes in Hp.
      rewrite <- (conts_ok_none _ _ _ Hconts Hl) in Hp.
      destruct (lookup (ac_continuation ac) ce) as [params | ] eqn:Hparams;
        [ | destruct i; discriminate Hp ].
      exact (apply_cont_args_flow _ _ _ _ _ Hvars He Hvs Hparams i v p Hv Hp).
  - (* [Switch] *)
    rewrite step_switch.
    destruct (eval_simple (state_vars s) scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    exists ce. simpl_state.
    split; [ | split; [ exact Hvars | split; [ exact Hconts | exact Hst ] ] ].
    exact (proj1 (every_in _ _ _) He (i, ac) (find_arm_compact _ _ _ Harm)).
  - (* [Invalid] *)
    rewrite step_invalid. exact I.
Qed.

Lemma run_flow_invariant (n : nat) (s : state) :
  flow_invariant s ->
  match run n s with
  | Step s' => flow_invariant s'
  | Exit k vs _ => exit_flows G sol u k vs
  | Undefined => True
  end.
Proof.
  revert s. induction n as [ | n IH]; intros s Hs; [ exact Hs | ].
  rewrite run_succ. pose proof (step_flow_invariant s Hs) as Hstep.
  destruct (step s) as [s' | k vs hp | ];
    [ exact (IH s' Hstep) | exact Hstep | exact I ].
Qed.

Lemma flow_invariant_flows (s : state) :
  flow_invariant s -> state_flows G sol s.
Proof.
  intros (ce & _ & Hvars & Hconts & Hst).
  split; [ exact Hvars | ].
  split; [ exact (conts_ok_flows _ _ Hconts) | exact Hst ].
Qed.

Lemma initial_flow_invariant (e : expr) :
  expr_edges G noted (unit_conts u) e ->
  flow_invariant (State e [] [] empty_static_env empty_heap).
Proof.
  intros He. exists (unit_conts u). simpl_state.
  split; [ exact He | split; [ | split; [ exact Conts_ok_nil | ] ] ].
  - intros x v Hx. discriminate Hx.
  - intros s allocated Hs. discriminate Hs.
Qed.

End Run.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

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

Theorem flow_sound : flow_sound_statement.
Proof.
  intros flag is_local u e T n Hlocal Hfrag Hcheck.
  pose proof (check_closed_sound is_local _ _ Hcheck) as Hc.
  destruct (traverse_unit_facts flag u e Hfrag)
    as (noted & He & Hconsts & Himported & _).
  set (G := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  (* Every symbol that went through [simple_to_node] is an allocation of
     its own: one that the unit binds has the facts of a block, and
     another any source. *)
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs.
    destruct (among s (static_names symbols_of_binding e)) eqn:Hunit.
    - apply among_In in Hunit.
      pose proof (proj1 (static_names_edges G noted) e _ He s Hunit) as Hcons.
      exact (alloc_self is_local G sol Hc _ _ _ Hcons
               (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
    - apply (may_come_from_self is_local G sol _ Hc).
      exact (any_source_input _ _ _ Hc _ (Himported s Hs Hunit)). }
  pose proof (run_flow_invariant is_local G sol noted u Hc Hlocal Hconsts
                Hsyms n _ (initial_flow_invariant G sol noted u e He))
    as Hrun.
  destruct (run n (State e [] [] empty_static_env empty_heap));
    [ exact (flow_invariant_flows _ _ _ _ _ Hrun) | exact Hrun | exact I ].
Qed.

Theorem unit_cont_nodes_used : unit_cont_nodes_used_statement.
Proof.
  intros flag u e k p Hfrag Hp.
  destruct (traverse_unit_facts flag u e Hfrag) as (_ & _ & _ & _ & Hused).
  exact (Hused k p Hp).
Qed.

Theorem example : example_statement.
Proof.
  split; [ intros f Hf; discriminate Hf | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ | vm_compute; reflexivity ].
  intros T Hin. vm_compute in Hin.
  repeat (destruct Hin as [<- | Hin]; [ vm_compute; reflexivity | ]).
  destruct Hin.
Qed.
