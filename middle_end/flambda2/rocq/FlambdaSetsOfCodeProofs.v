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


(* Proofs of the statements of FlambdaSetsOfCodeSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Arith Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaKnownSymbolsSpec FlambdaClosuresSpec
  FlambdaClosuresProofs FlambdaContextWhereSpec FlambdaKnownSlotsSpec
  FlambdaKnownSlotsProofs FlambdaSetsOfCodeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The test, of the sets of the parts of a term                             *)
(* ======================================================================== *)

Lemma sets_made_have_app (c : code_id) (f : function_slot) (fact : set_fact)
    (made made' : list set_made) :
  sets_made_have c f fact (made ++ made')
  = sets_made_have c f fact made && sets_made_have c f fact made'.
Proof. unfold sets_made_have. apply forallb_app. Qed.

Lemma bindings_sets_made_have (c : code_id) (f : function_slot)
    (fact : set_fact) (bindings : list static_binding) :
  sets_made_have c f fact (flat_map binding_sets_made bindings)
  = forallb
      (fun binding => sets_made_have c f fact (binding_sets_made binding))
      bindings.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite sets_made_have_app, IH. reflexivity.
Qed.

Lemma handlers_sets_made_have (c : code_id) (f : function_slot)
    (fact : set_fact)
    (handlers : list (continuation * continuation_handler)) :
  sets_made_have c f fact
    (flat_map (fun '(_, h) => handler_sets_made h) handlers)
  = forallb (fun '(_, h) => sets_made_have c f fact (handler_sets_made h))
      handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite sets_made_have_app, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* A set that a term makes, and the set of the closure that it gives        *)
(* ======================================================================== *)

(* The values of the value slots, in the order of the slots: a slot has
   the value of what the set puts in it. *)
Lemma eval_value_slots_lookup (vars : env) :
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) (slot : value_slot)
    (a : simple),
    eval_value_slots vars value_slots = Some slot_values ->
    lookup slot value_slots = Some a ->
    ∃ v : value,
      lookup slot slot_values = Some v ∧ eval_simple vars a = Some v.
Proof.
  induction value_slots as [ | [slot' s] value_slots IH];
    intros slot_values slot a; simpl.
  - intros _ Hlookup. discriminate Hlookup.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    destruct (Pos.eqb slot slot').
    + intros [= <-]. exists v. split; [ reflexivity | exact Hs ].
    + intros Hlookup. exact (IH rest slot a eq_refl Hlookup).
Qed.

(* A set that a term makes and that has the fact gives a closure whose
   set has it. *)
Lemma set_made_held (fact : set_fact) (vars : env)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) :
  fact_closed fact = true ->
  eval_value_slots vars value_slots = Some slot_values ->
  set_made_has fact (function_decls, value_slots) = true ->
  set_held_has fact function_decls slot_values = true.
Proof.
  intros Hclosed Heval.
  destruct fact as [slot code | slot held];
    cbn [set_made_has set_held_has fst snd fact_closed] in *;
    [ auto | ].
  destruct (lookup slot value_slots) as [a | ] eqn:Ha; [ | discriminate ].
  intros Hsame. apply same_simple_eq in Hsame. subst a.
  destruct (eval_value_slots_lookup vars value_slots slot_values slot held
              Heval Ha) as (v & -> & Hv).
  exact (eval_closed_value_is vars held v Hclosed Hv).
Qed.

Lemma set_made_held_as (c : code_id) (f : function_slot) (fact : set_fact)
    (vars : env) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) :
  fact_closed fact = true ->
  eval_value_slots vars value_slots = Some slot_values ->
  sets_made_have c f fact [(function_decls, value_slots)] = true ->
  set_held_as c f fact function_decls slot_values = true.
Proof.
  intros Hclosed Heval H. unfold sets_made_have in H. cbn [forallb fst] in H.
  rewrite andb_true_r in H. unfold set_held_as.
  apply orb_true_iff in H as [H | H]; [ rewrite H; reflexivity | ].
  rewrite (set_made_held fact vars function_decls value_slots slot_values
             Hclosed Heval H).
  apply orb_true_r.
Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Section Sets_have.

Variable c : code_id.
Variable f : function_slot.
Variable fact : set_fact.
Hypothesis Hclosed : fact_closed fact = true.

Let Pe (e : expr) : bool := sets_made_have c f fact (sets_made e).
Let Ph (h : continuation_handler) : bool :=
  sets_made_have c f fact (handler_sets_made h).
Let Pc (code : code) : bool := sets_made_have c f fact (code_sets_made code).

Let Pv (v : value) : Prop := value_sets_have c f fact v = true.
Let Qv (allocated : statically_allocated_value) : Prop :=
  allocated_sets_have c f fact allocated = true.

(* The terms. *)
Lemma sets_have_terms_inv_step (s s' : state) :
  terms_inv Pe Ph Pc s -> step s = Step s' -> terms_inv Pe Ph Pc s'.
Proof.
  apply terms_inv_step.
  - intros x kind d body H. unfold Pe in *. cbn [sets_made] in H.
    rewrite sets_made_have_app in H. apply andb_true_iff in H as [_ H].
    exact H.
  - intros bindings body H. unfold Pe in *. cbn [sets_made] in H.
    rewrite sets_made_have_app in H.
    apply andb_true_iff in H as [Hbindings Hbody].
    split; [ exact Hbody | ]. intros id code Hin.
    rewrite bindings_sets_made_have, forallb_forall in Hbindings.
    exact (Hbindings _ Hin).
  - intros k h body H. unfold Pe, Ph in *. cbn [sets_made] in H.
    rewrite sets_made_have_app in H. apply andb_true_iff in H. exact H.
  - intros handlers body H. unfold Pe, Ph in *. cbn [sets_made] in H.
    rewrite sets_made_have_app, handlers_sets_made_have in H.
    apply andb_true_iff in H. exact H.
  - intros ac. reflexivity.
  - intros params metadata handler H. exact H.
  - intros [params my_closure ret exn metadata body] H. exact H.
Qed.

(* The terms as FlambdaSetsOfCodeSpec.state_sets_have asks them. *)
Fixpoint cont_sets_have_terms (k : cont_value) {struct k} :
  cont_terms_ok Ph k <-> cont_sets_have c f fact k.
Proof.
  destruct k
    as [ h vars conts recursive | k0 | k' extra
       | remaining remaining_arity return_arity return_to exn_to ];
    cbn [cont_terms_ok cont_sets_have].
  - assert (Hconts :
      fold_right (fun kc acc => cont_terms_ok Ph (snd kc) ∧ acc) True conts
      <-> fold_right (fun kc acc => cont_sets_have c f fact (snd kc) ∧ acc)
            True conts).
    { induction conts as [ | [k1 c1] conts IH]; cbn [fold_right snd];
        [ reflexivity | ].
      rewrite (cont_sets_have_terms c1), IH. reflexivity. }
    rewrite Hconts. reflexivity.
  - reflexivity.
  - apply cont_sets_have_terms.
  - rewrite (cont_sets_have_terms return_to), (cont_sets_have_terms exn_to).
    reflexivity.
Qed.

Lemma conts_sets_have_terms (conts : cont_env) :
  conts_terms_ok Ph conts <-> conts_sets_have c f fact conts.
Proof.
  unfold conts_terms_ok, conts_sets_have.
  induction conts as [ | [k1 c1] conts IH]; cbn [fold_right snd];
    [ reflexivity | ].
  rewrite (cont_sets_have_terms c1), IH. reflexivity.
Qed.

Lemma state_sets_have_terms (s : state) :
  state_sets_have c f fact s
  <-> terms_inv Pe Ph Pc s
      ∧ state_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated)
          s.
Proof.
  unfold state_sets_have, terms_inv.
  rewrite (conts_sets_have_terms (state_conts s)).
  unfold code_terms_ok, code_sets_have, Pe, Pc, Pv, Qv. tauto.
Qed.

(* The values. *)
Lemma slot_values_sets_have_in (slot_values : list (value_slot * value)) :
  forallb (fun '(_, contents) => value_sets_have c f fact contents)
    slot_values
  = true ->
  ∀ (w : value_slot) (contents : value),
    (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros H w contents Hin. rewrite forallb_forall in H.
  exact (H (w, contents) Hin).
Qed.

Lemma statics_sets_have_lookup (statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  lookup s (static_values statics) = Some allocated ->
  Qv allocated.
Proof.
  intros Hstatics Hs. rewrite Forall_forall in Hstatics.
  exact (Hstatics (s, allocated) (lookup_in _ _ _ Hs)).
Qed.

Lemma eval_simple_sets_have (vars : env) (s : simple) (v : value) :
  env_values_all Pv vars -> eval_simple vars s = Some v -> Pv v.
Proof.
  intros Hvars. destruct s as [x | sym | k]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

Lemma eval_simples_sets_have (vars : env) :
  env_values_all Pv vars ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs ->
    forallb (value_sets_have c f fact) vs = true.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_sets_have vars s v Hvars Hs). exact (IH rest eq_refl).
Qed.

Lemma eval_value_slots_sets_have (vars : env) :
  env_values_all Pv vars ->
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)),
    eval_value_slots vars value_slots = Some slot_values ->
    forallb (fun '(_, contents) => value_sets_have c f fact contents)
      slot_values
    = true.
Proof.
  intros Hvars.
  induction value_slots as [ | [slot s] value_slots IH]; intros slot_values;
    simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_sets_have vars s v Hvars Hs). exact (IH rest eq_refl).
Qed.

Lemma values_sets_have_Forall (vs : list value) :
  forallb (value_sets_have c f fact) vs = true -> Forall Pv vs.
Proof.
  intros H. rewrite forallb_forall in H. apply Forall_forall. exact H.
Qed.

(* A closure that is so, read as a closure of a set: its set is so, and
   the values of its value slots are so. *)
Lemma closure_in_set_sets_have (statics : static_env) (v : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  Pv v ->
  closure_in_set statics v = Some (slot, decls, slot_values) ->
  set_held_as c f fact decls slot_values = true
  ∧ ∀ (w : value_slot) (contents : value),
      (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros Hstatics Hv. unfold closure_in_set.
  destruct v as [k | code slots | at_slot set_decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - unfold Pv in Hv. simpl in Hv. apply andb_true_iff in Hv as [Hset Hslots].
    intros [= <- <- <-].
    split; [ exact Hset | exact (slot_values_sets_have_in slots Hslots) ].
  - unfold Pv in Hv. simpl in Hv. apply andb_true_iff in Hv as [Hset Hslots].
    destruct (lookup at_slot set_decls); [ | discriminate ].
    simpl. intros [= <- <- <-].
    split; [ exact Hset | exact (slot_values_sets_have_in slots Hslots) ].
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms set_decls slots | tag fields | contents
          | bytes | elements | ] | ] eqn:Hs;
      try discriminate.
    + pose proof (statics_sets_have_lookup statics s _ Hstatics Hs) as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Hset Hslots].
      intros [= <- <- <-].
      split; [ exact Hset | exact (slot_values_sets_have_in slots Hslots) ].
    + pose proof (statics_sets_have_lookup statics s _ Hstatics Hs) as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Hset Hslots].
      destruct (lookup at_slot set_decls); [ | discriminate ].
      simpl. intros [= <- <- <-].
      split; [ exact Hset | exact (slot_values_sets_have_in slots Hslots) ].
Qed.

Lemma value_sets_have_of_numbers : holds_of_numbers Pv.
Proof. intros k. split; reflexivity. Qed.

Lemma value_sets_have_of_blocks : holds_of_blocks Pv.
Proof.
  intros tag fields.
  exact (forallb_forall (value_sets_have c f fact) fields).
Qed.

Lemma value_sets_have_of_arrays : holds_of_arrays Pv.
Proof.
  intros elements.
  exact (forallb_forall (value_sets_have c f fact) elements).
Qed.

Lemma value_sets_have_in_static_blocks (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_blocks statics Pv.
Proof.
  intros Hstatics s tag fields _ Hs v Hin.
  pose proof (statics_sets_have_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid. rewrite forallb_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma value_sets_have_in_static_arrays (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_arrays statics Pv.
Proof.
  intros Hstatics s elements _ Hs v Hin.
  pose proof (statics_sets_have_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid. rewrite forallb_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma project_function_slot_sets_have (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  Pv v -> project_function_slot statics move_from move_to v = Some w -> Pv w.
Proof.
  intros Hv. unfold project_function_slot.
  destruct v as [k | code slots | at_slot decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros Hproject.
    apply project_unnamed_function_slot in Hproject as (_ & _ & ->).
    exact Hv.
  - destruct (Pos.eqb at_slot move_from); [ | discriminate ].
    destruct (lookup at_slot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ].
    simpl. intros [= <-]. exact Hv.
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms decls slots | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + intros Hproject.
      apply project_unnamed_function_slot in Hproject as (_ & _ & ->).
      reflexivity.
    + destruct (Pos.eqb at_slot move_from); [ | discriminate ].
      destruct (lookup at_slot decls); [ | discriminate ].
      destruct (lookup move_to decls); [ | discriminate ].
      destruct (lookup move_to syms); [ | discriminate ].
      simpl. intros [= <-]. reflexivity.
Qed.

Lemma eval_named_sets_have (statics : static_env) (vars : env) (d : named)
    (v : value) :
  sets_made_have c f fact (named_sets_made d) = true ->
  env_values_all Pv vars ->
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  eval_named statics vars d = Some v ->
  Pv v.
Proof.
  intros Hd Hvars Hstatics.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    simpl.
  - exact (eval_simple_sets_have vars s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    simpl. intros Hprim.
    apply (eval_prim_preserves Pv statics p vs v value_sets_have_of_numbers
             value_sets_have_of_blocks value_sets_have_of_arrays
             (value_sets_have_in_static_blocks statics Hstatics)
             (value_sets_have_in_static_arrays statics Hstatics));
      [ | exact Hprim ].
    pose proof (eval_simples_sets_have vars Hvars args vs Hargs) as Hvs.
    rewrite forallb_forall in Hvs. exact Hvs.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl.
    cbn [named_sets_made] in Hd.
    rewrite (set_made_held_as c f fact vars _ slots vs Hclosed Hslots Hd).
    exact (eval_value_slots_sets_have vars Hvars slots vs Hslots).
  - destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl.
    cbn [named_sets_made] in Hd.
    rewrite (set_made_held_as c f fact vars _ slots vs Hclosed Hslots Hd).
    exact (eval_value_slots_sets_have vars Hvars slots vs Hslots).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    destruct (closure_of statics k) as [[code slot_values] | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics k code slot_values) Hclosure)
      as (at_slot & decls & Hin_set & _).
    destruct (closure_in_set_sets_have statics k at_slot decls slot_values
                Hstatics (eval_simple_sets_have vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics k) as [[[at_slot decls] slot_values] | ]
      eqn:Hin_set;
      [ | discriminate ].
    simpl. destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hlookup.
    destruct (closure_in_set_sets_have statics k at_slot decls slot_values
                Hstatics (eval_simple_sets_have vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    exact (project_function_slot_sets_have statics move_from move_to k v
             (eval_simple_sets_have vars closure k Hvars Hk)).
Qed.

Lemma eval_static_const_sets_have (const : static_const) (vars : env)
    (allocated : statically_allocated_value) :
  sets_made_have c f fact (static_const_sets_made const) = true ->
  env_values_all Pv vars ->
  eval_static_const vars const = Some allocated ->
  Qv allocated.
Proof.
  intros Hconst Hvars.
  destruct const as [code slots | slot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl.
    cbn [static_const_sets_made] in Hconst.
    rewrite (set_made_held_as c f fact vars _ slots vs Hclosed Hslots Hconst).
    exact (eval_value_slots_sets_have vars Hvars slots vs Hslots).
  - destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl.
    cbn [static_const_sets_made] in Hconst.
    rewrite (set_made_held_as c f fact vars _ slots vs Hclosed Hslots Hconst).
    exact (eval_value_slots_sets_have vars Hvars slots vs Hslots).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_sets_have vars Hvars fields vs Hfields).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simple_sets_have vars contents v Hvars Hcontents).
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_sets_have vars Hvars elements vs Helements).
Qed.

Lemma eval_static_bindings_sets_have (vars : env) :
  env_values_all Pv vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    forallb
      (fun binding => sets_made_have c f fact (binding_sets_made binding))
      bindings
    = true ->
    Forall (fun binding => Qv (snd binding)) (static_values statics) ->
    eval_static_bindings vars bindings statics = Some statics' ->
    Forall (fun binding => Qv (snd binding)) (static_values statics').
Proof.
  intros Hvars.
  induction bindings as [ | [id code | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ Hvalid [= <-]. exact Hvalid.
  - intros H Hvalid. apply andb_true_iff in H as [_ Hrest].
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply (IH _ _ Hrest). exact Hvalid.
  - intros Hrest Hvalid. exact (IH _ _ Hrest Hvalid).
  - intros H Hvalid. apply andb_true_iff in H as [Hconst Hrest].
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Heval;
      simpl; [ | discriminate ].
    apply (IH _ _ Hrest). simpl. apply Forall_cons; [ | exact Hvalid ].
    exact (eval_static_const_sets_have const vars allocated Hconst Hvars
             Heval).
Qed.

Lemma value_sets_have_partial (callee : value) (args : list value)
    (args_arity : arity) :
  Pv (Value_partial_application callee args args_arity)
  <-> Pv callee ∧ Forall Pv args.
Proof.
  unfold Pv. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity.
Qed.

Lemma values_sets_have_step (s : state) :
  Pe (state_expr s) = true ->
  state_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated) s ->
  match step s with
  | Step s' =>
    state_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated) s'
  | Exit _ vs _ _ => Forall Pv vs
  | Undefined => True
  end.
Proof.
  intros P Hvalid.
  apply (step_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated)
           s (fun _ => value_sets_have_partial) Hvalid).
  - intros size tuple fields Htuple Hfields.
    destruct Hvalid as (_ & _ & Hstatics & _).
    apply (tuple_fields_forall _ (state_statics s) size tuple fields);
      [ | exact Hfields ].
    intros field v Hprim.
    refine (eval_prim_preserves Pv (state_statics s) _ [tuple] v
              value_sets_have_of_numbers value_sets_have_of_blocks
              value_sets_have_of_arrays
              (value_sets_have_in_static_blocks (state_statics s) Hstatics)
              (value_sets_have_in_static_arrays (state_statics s) Hstatics) _
              Hprim).
    intros w [<- | []]. exact Htuple.
  - intros x kind d body v hp' rd' Hexpr Hexec. rewrite Hexpr in P.
    unfold Pe in P. cbn [sets_made] in P. rewrite sets_made_have_app in P.
    apply andb_true_iff in P as [Pd _].
    destruct Hvalid as (Hvars & _ & Hstatics & Hheap).
    refine
      (exec_named_values_all Pv (state_statics s) (state_heap s) hp'
         (state_reads s) rd'
              (state_vars s) d v value_sets_have_of_numbers
              value_sets_have_of_blocks value_sets_have_of_arrays
              (value_sets_have_in_static_blocks (state_statics s) Hstatics)
              (value_sets_have_in_static_arrays (state_statics s) Hstatics)
              (fun _ => eq_refl) Hheap _ _ Hexec).
    + intros p args vs _ Hvs.
      exact (proj1 (forallb_forall _ _)
               (eval_simples_sets_have (state_vars s) Hvars args vs Hvs)).
    + intros _ w Heval.
      exact (eval_named_sets_have (state_statics s) (state_vars s) d w Pd
               Hvars Hstatics Heval).
  - intros bindings body statics' Hexpr Heval.
    split; [ exact (fun v H => H) | ].
    rewrite Hexpr in P. unfold Pe in P. cbn [sets_made] in P.
    rewrite sets_made_have_app, bindings_sets_made_have in P.
    apply andb_true_iff in P as [Pbindings _].
    destruct Hvalid as (Hvars & _ & Hstatics & _).
    exact (eval_static_bindings_sets_have (state_vars s) Hvars bindings
             (state_statics s) statics' Pbindings Hstatics Heval).
  - intros ap callee vs extra _ Hcallee Hargs Hextra.
    destruct Hvalid as (Hvars & _).
    split; [ | split ].
    + exact (eval_simple_sets_have (state_vars s) (ap_callee ap) callee Hvars
               Hcallee).
    + exact (values_sets_have_Forall vs
               (eval_simples_sets_have (state_vars s) Hvars (ap_args ap) vs
                  Hargs)).
    + exact (values_sets_have_Forall extra
               (eval_simples_sets_have (state_vars s) Hvars
                  (exn_extra_simples ap) extra Hextra)).
  - intros ac vs _ Hargs. destruct Hvalid as (Hvars & _).
    exact (values_sets_have_Forall vs
             (eval_simples_sets_have (state_vars s) Hvars (ac_args ac) vs
                Hargs)).
Qed.

Lemma state_sets_have_kept : kept_by_step (state_sets_have c f fact).
Proof.
  intros s s' Hinv Hstep.
  apply state_sets_have_terms in Hinv as [Hterms Hvalues].
  apply state_sets_have_terms. split.
  - exact (sets_have_terms_inv_step s s' Hterms Hstep).
  - pose proof (values_sets_have_step s (proj1 Hterms) Hvalues) as H.
    rewrite Hstep in H. exact H.
Qed.

Lemma closure_set_has_aux (s : state) (x : variable) (v : value)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  state_sets_have c f fact s ->
  lookup x (state_vars s) = Some v ->
  closure_in_set (state_statics s) v = Some (f, function_decls, slot_values) ->
  lookup f function_decls = Some c ->
  set_held_has fact function_decls slot_values = true.
Proof.
  intros (_ & _ & _ & Hvars & _ & Hstatics & _) Hx Hin_set Hcode.
  destruct (closure_in_set_sets_have (state_statics s) v f function_decls
              slot_values Hstatics
              (env_values_all_lookup _ (state_vars s) x v Hvars Hx) Hin_set)
    as [Hset _].
  unfold set_held_as, has_code_at in Hset. rewrite Hcode, Pos.eqb_refl in Hset.
  exact Hset.
Qed.

End Sets_have.

Theorem state_sets_have_from : state_sets_have_from_statement.
Proof.
  intros c f fact e conts statics He Hconts Hvalues Hcode Hstatics.
  unfold state_sets_have, state_values_all. simpl_state.
  refine (conj He (conj Hconts (conj Hcode _))).
  exact
    (conj (Forall_nil _)
       (conj Hvalues (conj Hstatics (heap_values_all_empty _)))).
Qed.

Theorem state_sets_have_initial : state_sets_have_initial_statement.
Proof.
  intros c f fact e He.
  apply state_sets_have_from; [ exact He | exact I | exact I | | ].
  - intros id code Hlookup. discriminate Hlookup.
  - apply Forall_nil.
Qed.

Theorem state_sets_have_step : state_sets_have_step_statement.
Proof.
  intros c f fact Hclosed. exact (state_sets_have_kept c f fact Hclosed).
Qed.

Theorem closure_set_has : closure_set_has_statement.
Proof.
  intros c f fact s x v function_decls slot_values.
  apply closure_set_has_aux.
Qed.
