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

(* Proofs of the statements of FlambdaSlotsThroughSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Arith Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaStaticsInScopeSpec
  FlambdaNormaliseSpec
  FlambdaUnusedStaticsSpec FlambdaClosuresProofs FlambdaContextWhereSpec
  FlambdaContextWhereProofs
  FlambdaKnownSlotsSpec FlambdaSetsOfCodeSpec FlambdaSetsOfCodeProofs
  FlambdaSlotsOfCodeSpec FlambdaSlotsOfCodeProofs FlambdaImportsSpec
  FlambdaObserversSpec
  FlambdaUnitFromProofs FlambdaClosureOfCodeSpec FlambdaClosureOfCodeProofs
  FlambdaSlotsThroughSpec.
From OxCamlFlambda2 Require Import FlambdaBindersSpec FlambdaScopedProofs
  FlambdaScopeTactic FlambdaNominalProofs FlambdaEquivarianceProofs
  FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* What is in sight is kept where a variable is bound                       *)
(* ======================================================================== *)

Lemma eval_simple_bound_other (x : variable) (v : value) (vars : env)
    (s : simple) :
  var_in_simple x s = false ->
  eval_simple ((x, v) :: vars) s = eval_simple vars s.
Proof.
  destruct s as [y | sym | c]; simpl; try reflexivity.
  intros Hx. rewrite Pos.eqb_sym, Hx. reflexivity.
Qed.

Lemma lookup_forget_in_made (x : variable) (made : made_seen)
    (slot : value_slot) (s : simple) :
  lookup slot (forget_in_made x made) = Some (Some s) ->
  lookup slot made = Some (Some s) ∧ var_in_simple x s = false.
Proof.
  unfold forget_in_made. rewrite lookup_map_values.
  destruct (lookup slot made) as [[s0 | ] | ]; simpl; try discriminate.
  destruct (var_in_simple x s0) eqn:Hin; try discriminate.
  intros H. injection H as <-. split; [reflexivity | exact Hin].
Qed.

Lemma made_seen_holds_bound (x : variable) (v : value) (vars : env)
    (made : made_seen) (slot_values : list (value_slot * value)) :
  (∀ (slot : value_slot) (s : simple),
     lookup slot made = Some (Some s) ->
     ∃ w : value,
       lookup slot slot_values = Some w ∧ eval_simple vars s = Some w) ->
  ∀ (slot : value_slot) (s : simple),
    lookup slot (forget_in_made x made) = Some (Some s) ->
    ∃ w : value,
      lookup slot slot_values = Some w
      ∧ eval_simple ((x, v) :: vars) s = Some w.
Proof.
  intros Hmade slot s Hslot.
  destruct (lookup_forget_in_made _ _ _ _ Hslot) as [Hs Hin].
  destruct (Hmade _ _ Hs) as [w [Hw Heval]].
  exists w. split; [exact Hw | ].
  rewrite eval_simple_bound_other by exact Hin. exact Heval.
Qed.

Lemma made_sight_holds_without (known : made_sight) (vars : env)
    (x : variable) (v : value) :
  made_sight_holds known vars ->
  made_sight_holds (made_without x known) ((x, v) :: vars).
Proof.
  intros Hholds y made Hy. unfold made_without in Hy.
  rewrite lookup_map_values, lookup_unbind in Hy.
  destruct (Pos.eqb y x) eqn:Hyx; [discriminate | ].
  destruct (lookup y known) as [made0 | ] eqn:Hknown; [ | discriminate].
  simpl in Hy. injection Hy as <-.
  destruct (Hholds _ _ Hknown) as [vy [slot_values [Hvy [Hslots Hmade]]]].
  exists vy, slot_values. split; [ | split].
  - simpl. rewrite Hyx. exact Hvy.
  - exact Hslots.
  - apply made_seen_holds_bound. exact Hmade.
Qed.

Lemma made_sight_holds_nil (vars : env) : made_sight_holds [] vars.
Proof. intros y made Hy. discriminate. Qed.

(* The parameters of a handler, bound as the machine binds them. *)
Lemma made_sight_holds_bind_params (params : list variable) :
  ∀ (known : made_sight) (vs : list value) (vars handler_vars : env),
    made_sight_holds known vars ->
    bind_params params vs vars = Some handler_vars ->
    made_sight_holds (made_without_all params known) handler_vars.
Proof.
  induction params as [ | x params IH];
    intros known vs vars handler_vars Hholds Hbind.
  - destruct vs; [ | discriminate]. simpl in Hbind. injection Hbind as <-.
    exact Hholds.
  - destruct vs as [ | v vs]; [discriminate | ]. simpl in Hbind |- *.
    exact (IH _ _ _ _ (made_sight_holds_without _ _ x v Hholds) Hbind).
Qed.

Lemma eval_value_slots_first (vars : env) :
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) (slot : value_slot)
    (s : simple),
    eval_value_slots vars value_slots = Some slot_values ->
    lookup slot value_slots = Some s ->
    ∃ w : value,
      lookup slot slot_values = Some w ∧ eval_simple vars s = Some w.
Proof.
  induction value_slots as [ | [slot0 s0] value_slots IH];
    intros slot_values slot s Heval Hslot; [discriminate | ].
  simpl in Heval.
  destruct (eval_simple vars s0) as [w0 | ] eqn:Hs0; [ | discriminate].
  destruct (eval_value_slots vars value_slots) as [rest | ] eqn:Hrest;
    [ | discriminate].
  simpl in Heval. injection Heval as <-. simpl in Hslot |- *.
  destruct (Pos.eqb slot slot0).
  - injection Hslot as <-. exists w0. split; [reflexivity | exact Hs0].
  - exact (IH _ _ _ eq_refl Hslot).
Qed.

Lemma made_new_holds (vars : env) (x : variable) (v : value)
    (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) :
  eval_value_slots vars value_slots = Some slot_values ->
  ∀ (slot : value_slot) (s : simple),
    lookup slot (forget_in_made x (map_values Some value_slots))
    = Some (Some s) ->
    ∃ w : value,
      lookup slot slot_values = Some w
      ∧ eval_simple ((x, v) :: vars) s = Some w.
Proof.
  intros Heval. apply made_seen_holds_bound. intros slot s Hslot.
  rewrite lookup_map_values in Hslot.
  destruct (lookup slot value_slots) as [s0 | ] eqn:Hs0; [ | discriminate].
  simpl in Hslot. injection Hslot as <-.
  exact (eval_value_slots_first _ _ _ _ _ Heval Hs0).
Qed.

Theorem made_sight_holds_under_let : made_sight_holds_under_let_statement.
Proof.
  intros statics known vars x d v Hholds Heval.
  assert (Hwithout := made_sight_holds_without known vars x v Hholds).
  unfold made_under_let.
  destruct d as [s | prim args | code value_slots
                | slot function_decls value_slots | slot closure
                | project_from slot closure | move_from move_to closure];
    simpl; try exact Hwithout.
  - (* Set_of_closures *)
    simpl in Heval.
    destruct (eval_value_slots vars value_slots) as [slot_values | ] eqn:Hvs;
      [ | discriminate].
    simpl in Heval. injection Heval as <-.
    intros y made Hy. simpl in Hy.
    destruct (Pos.eqb y x) eqn:Hyx.
    + injection Hy as <-. apply Pos.eqb_eq in Hyx. subst y.
      exists (Value_closure code slot_values), slot_values.
      split; [simpl; rewrite Pos.eqb_refl; reflexivity | ].
      split; [reflexivity | ].
      apply made_new_holds. exact Hvs.
    + apply Hwithout. simpl. exact Hy.
  - (* Closure_of_set *)
    simpl in Heval.
    destruct (lookup slot function_decls) as [c | ]; [ | discriminate].
    simpl in Heval.
    destruct (eval_value_slots vars value_slots) as [slot_values | ] eqn:Hvs;
      [ | discriminate].
    simpl in Heval. injection Heval as <-.
    intros y made Hy. simpl in Hy.
    destruct (Pos.eqb y x) eqn:Hyx.
    + injection Hy as <-. apply Pos.eqb_eq in Hyx. subst y.
      exists (Value_closure_of_set slot function_decls slot_values),
        slot_values.
      split; [simpl; rewrite Pos.eqb_refl; reflexivity | ].
      split; [reflexivity | ].
      apply made_new_holds. exact Hvs.
    + apply Hwithout. simpl. exact Hy.
Qed.

(* What a step of a [Let] binds, a primitive included: what a primitive
   gives is no closure that is in sight. *)
Lemma made_sight_holds_exec (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (known : made_sight) (vars : env) (x : variable)
    (defining : named) (v : value) :
  made_sight_holds known vars ->
  exec_named statics hp rd vars defining = Some (v, hp', rd') ->
  made_sight_holds (made_under_let known x defining) ((x, v) :: vars).
Proof.
  intros Hholds Hexec.
  assert (Hcases :
            (∃ (p : prim) (args : list simple), defining = Prim p args)
            ∨ (∀ (p : prim) (args : list simple), defining <> Prim p args)).
  { destruct defining; try (right; intros ? ?; discriminate).
    left. eexists. eexists. reflexivity. }
  destruct Hcases as [(p & args & ->) | Hnot].
  - exact (made_sight_holds_without known vars x v Hholds).
  - rewrite (exec_named_not_prim _ _ rd_ _ _ Hnot) in Hexec.
    destruct (eval_named statics vars defining) as [w | ] eqn:Heval;
      [ | discriminate ].
    simpl in Hexec. injection Hexec as <- _ _.
    exact (made_sight_holds_under_let statics known vars x defining w Hholds
             Heval).
Qed.

(* ======================================================================== *)
(* The test, of the sets of the parts of a term                             *)
(* ======================================================================== *)

Lemma sets_through_give_app (c : code_id) (f : function_slot)
    (a b d : value_slot) (seen seen' : list set_through) :
  sets_through_give c f a b d (seen ++ seen')
  = sets_through_give c f a b d seen && sets_through_give c f a b d seen'.
Proof. unfold sets_through_give. apply forallb_app. Qed.

Lemma bindings_sets_through_give (c : code_id) (f : function_slot)
    (a b d : value_slot) (known : made_sight)
    (bindings : list static_binding) :
  sets_through_give c f a b d (flat_map (binding_sets_through known) bindings)
  = forallb
      (fun binding =>
         sets_through_give c f a b d (binding_sets_through known binding))
      bindings.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite sets_through_give_app, IH. reflexivity.
Qed.

Lemma handlers_sets_through_give (c : code_id) (f : function_slot)
    (a b d : value_slot) (known : made_sight)
    (handlers : list (continuation * continuation_handler)) :
  sets_through_give c f a b d
    (flat_map (fun '(_, h) => handler_sets_through known h) handlers)
  = forallb
      (fun '(_, h) =>
         sets_through_give c f a b d (handler_sets_through known h))
      handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite sets_through_give_app, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* A set that a term makes, and the set of the closure that it gives        *)
(* ======================================================================== *)

Lemma lookup_slots_through (known : made_sight) (slot : value_slot) :
  ∀ value_slots : list (value_slot * simple),
    lookup slot (slots_through known value_slots)
    = match lookup slot value_slots with
      | Some s => Some (s, simple_made_seen known s)
      | None => None
      end.
Proof.
  induction value_slots as [ | [slot' s] value_slots IH]; simpl;
    [ reflexivity | ].
  destruct (Pos.eqb slot slot'); [ reflexivity | exact IH ].
Qed.

Lemma same_simple_eq (s s' : simple) : same_simple s s' = true -> s = s'.
Proof.
  unfold same_simple.
  destruct (FlambdaModel.simple_eq_dec s s') as [Heq | _];
    [ intros _; exact Heq | intros H; discriminate H ].
Qed.

(* A set that a term makes under a sight that holds, and that passes
   the test, gives a closure whose set is so. *)
Lemma set_through_held (c : code_id) (f : function_slot) (a b d : value_slot)
    (known : made_sight) (vars : env)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) :
  made_sight_holds known vars ->
  eval_value_slots vars value_slots = Some slot_values ->
  sets_through_give c f a b d
    [(function_decls, slots_through known value_slots)]
  = true ->
  set_held_through c f a b d function_decls slot_values.
Proof.
  intros Hholds Heval H Hcode. unfold sets_through_give in H.
  cbn [forallb fst] in H. rewrite andb_true_r, Hcode in H. simpl in H.
  unfold set_through_gives in H. cbn [snd] in H.
  rewrite !lookup_slots_through in H.
  destruct (lookup a value_slots) as [sa | ] eqn:Ha; [ | discriminate ].
  destruct (simple_made_seen known sa) as [made | ] eqn:Hmade;
    [ | discriminate ].
  destruct (lookup b value_slots) as [held | ] eqn:Hb; [ | discriminate ].
  destruct (lookup d made) as [[inner | ] | ] eqn:Hd; try discriminate.
  apply same_simple_eq in H. subst inner.
  destruct sa as [y | sym | k]; simpl in Hmade; try discriminate.
  destruct (Hholds y made Hmade) as (v & inner_slots & Hy & Hclosure & Hslots).
  destruct (eval_value_slots_first vars value_slots slot_values a (Var y)
              Heval Ha) as (wa & Hwa & Hva).
  simpl in Hva. rewrite Hy in Hva. injection Hva as <-.
  destruct (eval_value_slots_first vars value_slots slot_values b held Heval
              Hb) as (wb & Hwb & Hvb).
  destruct (Hslots d held Hd) as (w & Hw & Hvw).
  rewrite Hvb in Hvw. injection Hvw as <-.
  exists v, inner_slots, wb. repeat split; assumption.
Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Lemma fold_right_and_Forall {A : Type} (P : A -> Prop) (l : list A) :
  fold_right (fun x acc => P x ∧ acc) True l <-> Forall P l.
Proof.
  induction l as [ | x l IH]; simpl.
  - split; [ intros _; apply Forall_nil | intros _; exact I ].
  - rewrite IH. split.
    + intros [Hx Hl]. apply Forall_cons; assumption.
    + intros H. inversion H; subst. split; assumption.
Qed.

Section Slots_through.

Variable c : code_id.
Variable f : function_slot.
Variable a b d : value_slot.

Let Pv (v : value) : Prop := value_slots_through c f a b d v.
Let Qv (allocated : statically_allocated_value) : Prop :=
  allocated_slots_through c f a b d allocated.

(* The values. *)
Lemma slots_through_Forall (slot_values : list (value_slot * value)) :
  fold_right
    (fun contents acc => value_slots_through c f a b d (snd contents) ∧ acc)
    True slot_values
  <-> Forall (fun contents => Pv (snd contents)) slot_values.
Proof.
  exact (fold_right_and_Forall (fun contents => Pv (snd contents))
           slot_values).
Qed.

Lemma values_through_Forall (vs : list value) :
  values_slots_through c f a b d vs <-> Forall Pv vs.
Proof. exact (fold_right_and_Forall Pv vs). Qed.

Lemma values_through_map_snd (slot_values : list (value_slot * value)) :
  values_slots_through c f a b d (map snd slot_values)
  <-> Forall (fun contents => Pv (snd contents)) slot_values.
Proof. rewrite values_through_Forall. apply Forall_map. Qed.

Lemma slot_values_through_in (slot_values : list (value_slot * value)) :
  Forall (fun contents => Pv (snd contents)) slot_values ->
  ∀ (w : value_slot) (contents : value),
    (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros H w contents Hin. rewrite Forall_forall in H.
  exact (H (w, contents) Hin).
Qed.

Lemma statics_through_lookup (statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  lookup s (static_values statics) = Some allocated ->
  Qv allocated.
Proof.
  intros Hstatics Hs. rewrite Forall_forall in Hstatics.
  exact (Hstatics (s, allocated) (lookup_in _ _ _ Hs)).
Qed.

Lemma eval_simple_through (vars : env) (s : simple) (v : value) :
  env_values_all Pv vars -> eval_simple vars s = Some v -> Pv v.
Proof.
  intros Hvars. destruct s as [x | sym | k]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. exact I.
  - intros [= <-]. exact I.
Qed.

Lemma eval_simples_through (vars : env) :
  env_values_all Pv vars ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs -> Forall Pv vs.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. apply Forall_nil.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-].
    apply Forall_cons; [ | exact (IH rest eq_refl) ].
    exact (eval_simple_through vars s v Hvars Hs).
Qed.

Lemma eval_value_slots_through (vars : env) :
  env_values_all Pv vars ->
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)),
    eval_value_slots vars value_slots = Some slot_values ->
    Forall (fun contents => Pv (snd contents)) slot_values.
Proof.
  intros Hvars.
  induction value_slots as [ | [slot' s] value_slots IH]; intros slot_values;
    simpl.
  - intros [= <-]. apply Forall_nil.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-].
    apply Forall_cons; [ | exact (IH rest eq_refl) ].
    exact (eval_simple_through vars s v Hvars Hs).
Qed.

(* A closure that is so, read as a closure of a set: its set is so, and
   the values of its value slots are so. *)
Lemma closure_in_set_through (statics : static_env) (v : value)
    (at_slot : function_slot) (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  Pv v ->
  closure_in_set statics v = Some (at_slot, decls, slot_values) ->
  set_held_through c f a b d decls slot_values
  ∧ ∀ (w : value_slot) (contents : value),
      (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros Hstatics Hv. unfold closure_in_set.
  destruct v as [k | code slots | at_slot' set_decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - unfold Pv in Hv. simpl in Hv. destruct Hv as [Hset Hslots].
    apply slots_through_Forall in Hslots.
    intros [= <- <- <-].
    split; [ exact Hset | exact (slot_values_through_in slots Hslots) ].
  - unfold Pv in Hv. simpl in Hv. destruct Hv as [Hset Hslots].
    apply slots_through_Forall in Hslots.
    destruct (lookup at_slot' set_decls); [ | discriminate ].
    simpl. intros [= <- <- <-].
    split; [ exact Hset | exact (slot_values_through_in slots Hslots) ].
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot' syms set_decls slots | tag fields | contents
          | bytes | elements | ] | ] eqn:Hs;
      try discriminate.
    + pose proof (statics_through_lookup statics s _ Hstatics Hs) as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid. destruct Hvalid as [Hset Hslots].
      apply values_through_map_snd in Hslots.
      intros [= <- <- <-].
      split; [ exact Hset | exact (slot_values_through_in slots Hslots) ].
    + pose proof (statics_through_lookup statics s _ Hstatics Hs) as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid. destruct Hvalid as [Hset Hslots].
      apply values_through_map_snd in Hslots.
      destruct (lookup at_slot' set_decls); [ | discriminate ].
      simpl. intros [= <- <- <-].
      split; [ exact Hset | exact (slot_values_through_in slots Hslots) ].
Qed.

Lemma value_through_of_numbers : holds_of_numbers Pv.
Proof. intros k. split; exact I. Qed.

Lemma value_through_of_blocks : holds_of_blocks Pv.
Proof.
  intros tag fields. unfold Pv. simpl.
  rewrite (fold_right_and_Forall (value_slots_through c f a b d) fields).
  apply Forall_forall.
Qed.

Lemma value_through_of_arrays : holds_of_arrays Pv.
Proof.
  intros elements. unfold Pv. simpl.
  rewrite (fold_right_and_Forall (value_slots_through c f a b d) elements).
  apply Forall_forall.
Qed.

Lemma value_through_in_static_blocks (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_blocks statics Pv.
Proof.
  intros Hstatics s tag fields _ Hs v Hin.
  pose proof (statics_through_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid.
  apply values_through_Forall in Hvalid. rewrite Forall_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma value_through_in_static_arrays (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_arrays statics Pv.
Proof.
  intros Hstatics s elements _ Hs v Hin.
  pose proof (statics_through_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid.
  apply values_through_Forall in Hvalid. rewrite Forall_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma project_function_slot_through (statics : static_env)
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
      exact I.
    + destruct (Pos.eqb at_slot move_from); [ | discriminate ].
      destruct (lookup at_slot decls); [ | discriminate ].
      destruct (lookup move_to decls); [ | discriminate ].
      destruct (lookup move_to syms); [ | discriminate ].
      simpl. intros [= <-]. exact I.
Qed.

Lemma eval_named_through (statics : static_env) (known : made_sight)
    (vars : env) (defining : named) (v : value) :
  made_sight_holds known vars ->
  sets_through_give c f a b d (named_sets_through known defining) = true ->
  env_values_all Pv vars ->
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  eval_named statics vars defining = Some v ->
  Pv v.
Proof.
  intros Hhold Hd Hvars Hstatics.
  destruct defining as [s | p args | code slots | at_slot decls slots
    | slot' closure | project_from slot' closure
    | move_from move_to closure];
    simpl.
  - exact (eval_simple_through vars s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    simpl. intros Hprim.
    apply (eval_prim_preserves Pv statics p vs v value_through_of_numbers
             value_through_of_blocks value_through_of_arrays
             (value_through_in_static_blocks statics Hstatics)
             (value_through_in_static_arrays statics Hstatics));
      [ | exact Hprim ].
    pose proof (eval_simples_through vars Hvars args vs Hargs) as Hvs.
    rewrite Forall_forall in Hvs. exact Hvs.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl.
    cbn [named_sets_through] in Hd.
    split.
    + exact (set_through_held c f a b d known vars _ slots vs Hhold Hslots Hd).
    + apply slots_through_Forall.
      exact (eval_value_slots_through vars Hvars slots vs Hslots).
  - destruct (lookup at_slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl.
    cbn [named_sets_through] in Hd.
    split.
    + exact (set_through_held c f a b d known vars _ slots vs Hhold Hslots Hd).
    + apply slots_through_Forall.
      exact (eval_value_slots_through vars Hvars slots vs Hslots).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    destruct (closure_of statics k) as [[code slot_values] | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics k code slot_values) Hclosure)
      as (at_slot & decls & Hin_set & _).
    destruct (closure_in_set_through statics k at_slot decls slot_values
                Hstatics (eval_simple_through vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot' v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics k) as [[[at_slot decls] slot_values] | ]
      eqn:Hin_set;
      [ | discriminate ].
    simpl. destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hlookup.
    destruct (closure_in_set_through statics k at_slot decls slot_values
                Hstatics (eval_simple_through vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot' v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    exact (project_function_slot_through statics move_from move_to k v
             (eval_simple_through vars closure k Hvars Hk)).
Qed.

Lemma eval_static_const_through (known : made_sight) (const : static_const)
    (vars : env) (allocated : statically_allocated_value) :
  made_sight_holds known vars ->
  sets_through_give c f a b d (static_const_sets_through known const) = true ->
  env_values_all Pv vars ->
  eval_static_const vars const = Some allocated ->
  Qv allocated.
Proof.
  intros Hhold Hconst Hvars.
  destruct const as [code slots | at_slot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl.
    cbn [static_const_sets_through] in Hconst.
    split.
    + exact (set_through_held c f a b d known vars _ slots vs Hhold Hslots
               Hconst).
    + apply values_through_map_snd.
      exact (eval_value_slots_through vars Hvars slots vs Hslots).
  - destruct (lookup at_slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl.
    cbn [static_const_sets_through] in Hconst.
    split.
    + exact (set_through_held c f a b d known vars _ slots vs Hhold Hslots
               Hconst).
    + apply values_through_map_snd.
      exact (eval_value_slots_through vars Hvars slots vs Hslots).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl. apply values_through_Forall.
    exact (eval_simples_through vars Hvars fields vs Hfields).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simple_through vars contents v Hvars Hcontents).
  - intros [= <-]. exact I.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl. apply values_through_Forall.
    exact (eval_simples_through vars Hvars elements vs Helements).
Qed.

Lemma eval_static_bindings_through (known : made_sight) (vars : env) :
  made_sight_holds known vars ->
  env_values_all Pv vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    forallb
      (fun binding =>
         sets_through_give c f a b d (binding_sets_through known binding))
      bindings
    = true ->
    Forall (fun binding => Qv (snd binding)) (static_values statics) ->
    eval_static_bindings vars bindings statics = Some statics' ->
    Forall (fun binding => Qv (snd binding)) (static_values statics').
Proof.
  intros Hhold Hvars.
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
    exact (eval_static_const_through known const vars allocated Hhold Hconst
             Hvars Heval).
Qed.

Lemma value_through_partial (callee : value) (args : list value)
    (args_arity : arity) :
  Pv (Value_partial_application callee args args_arity)
  <-> Pv callee ∧ Forall Pv args.
Proof.
  unfold Pv. simpl.
  rewrite (fold_right_and_Forall (value_slots_through c f a b d) args).
  reflexivity.
Qed.

Lemma values_through_step (s : state) :
  expr_slots_through c f a b d (state_vars s) (state_expr s) ->
  state_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated) s ->
  match step s with
  | Step s' =>
    state_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated) s'
  | Exit _ vs _ _ => Forall Pv vs
  | Undefined => True
  end.
Proof.
  intros (known & Hhold & P) Hvalid.
  apply (step_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated)
           s (fun _ => value_through_partial) Hvalid).
  - intros size tuple fields Htuple Hfields.
    destruct Hvalid as (_ & _ & Hstatics & _).
    apply (tuple_fields_forall _ (state_statics s) size tuple fields);
      [ | exact Hfields ].
    intros field v Hprim.
    refine (eval_prim_preserves Pv (state_statics s) _ [tuple] v
              value_through_of_numbers value_through_of_blocks
              value_through_of_arrays
              (value_through_in_static_blocks (state_statics s) Hstatics)
              (value_through_in_static_arrays (state_statics s) Hstatics) _
              Hprim).
    intros w [<- | []]. exact Htuple.
  - intros x kind defining body v hp' rd' Hexpr Hexec. rewrite Hexpr in P.
    cbn [sets_through] in P. rewrite sets_through_give_app in P.
    apply andb_true_iff in P as [Pd _].
    destruct Hvalid as (Hvars & _ & Hstatics & Hheap).
    refine
      (exec_named_values_all Pv (state_statics s) (state_heap s) hp'
         (state_reads s) rd'
              (state_vars s) defining v value_through_of_numbers
              value_through_of_blocks value_through_of_arrays
              (value_through_in_static_blocks (state_statics s) Hstatics)
              (value_through_in_static_arrays (state_statics s) Hstatics)
              (fun _ => I) Hheap _ _ Hexec).
    + intros p args vs _ Hvs.
      exact (proj1 (Forall_forall _ _)
               (eval_simples_through (state_vars s) Hvars args vs Hvs)).
    + intros _ w Heval.
      exact (eval_named_through (state_statics s) known (state_vars s)
               defining w Hhold Pd Hvars Hstatics Heval).
  - intros bindings body statics' Hexpr Heval.
    split; [ exact (fun v H => H) | ].
    rewrite Hexpr in P. cbn [sets_through] in P.
    rewrite sets_through_give_app, bindings_sets_through_give in P.
    apply andb_true_iff in P as [Pbindings _].
    destruct Hvalid as (Hvars & _ & Hstatics & _).
    exact (eval_static_bindings_through known (state_vars s) Hhold Hvars
             bindings (state_statics s) statics' Pbindings Hstatics Heval).
  - intros ap callee vs extra _ Hcallee Hargs Hextra.
    destruct Hvalid as (Hvars & _).
    split; [ | split ].
    + exact (eval_simple_through (state_vars s) (ap_callee ap) callee Hvars
               Hcallee).
    + exact (eval_simples_through (state_vars s) Hvars (ap_args ap) vs Hargs).
    + exact (eval_simples_through (state_vars s) Hvars (exn_extra_simples ap)
               extra Hextra).
  - intros ac vs _ Hargs. destruct Hvalid as (Hvars & _).
    exact (eval_simples_through (state_vars s) Hvars (ac_args ac) vs Hargs).
Qed.

(* The terms. *)
Definition terms_through (s : state) : Prop :=
  expr_slots_through c f a b d (state_vars s) (state_expr s)
  ∧ conts_slots_through c f a b d (state_conts s)
  ∧ code_slots_through c f a b d (state_statics s).

Lemma conts_slots_through_app (conts conts' : cont_env) :
  conts_slots_through c f a b d conts ->
  conts_slots_through c f a b d conts' ->
  conts_slots_through c f a b d (conts ++ conts').
Proof.
  unfold conts_slots_through.
  induction conts as [ | kc conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] H'. split; [ exact Hc | exact (IH Hrest H') ].
Qed.

Lemma resolve_slots_through (k : continuation) (conts : cont_env) :
  conts_slots_through c f a b d conts ->
  cont_slots_through c f a b d (resolve k conts).
Proof.
  unfold resolve, conts_slots_through.
  induction conts as [ | [k' cv] conts IH]; simpl; [ intros _; exact I | ].
  intros [Hc Hrest]. destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma rec_closures_slots_through (known : made_sight) (vars : env)
    (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  made_sight_holds known vars ->
  forallb
    (fun '(_, h) =>
       sets_through_give c f a b d (handler_sets_through known h))
    all
  = true ->
  conts_slots_through c f a b d conts ->
  forallb
    (fun '(_, h) =>
       sets_through_give c f a b d (handler_sets_through known h))
    handlers
  = true ->
  conts_slots_through c f a b d
    (map (fun '(k, h) => (k, Closure h vars conts all)) handlers).
Proof.
  intros Hhold Hall Hconts. unfold conts_slots_through.
  induction handlers as [ | [k h] handlers IH]; simpl;
    [ intros _; exact I | ].
  intros H. apply andb_true_iff in H as [Hh Hrest].
  split; [ | exact (IH Hrest) ].
  split; [ | exact Hconts ].
  exists known. exact (conj Hhold (conj Hh Hall)).
Qed.

Lemma with_extra_args_slots_through (cv : cont_value) (extra : list value) :
  cont_slots_through c f a b d cv ->
  cont_slots_through c f a b d (with_extra_args cv extra).
Proof. destruct extra; intros Hc; exact Hc. Qed.

Lemma eval_static_bindings_code_slots_through (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    (∀ (id : code_id) (code : code),
      Bind_code id code ∈ bindings ->
      sets_through_give c f a b d (code_sets_through code) = true) ->
    code_slots_through c f a b d statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    code_slots_through c f a b d statics'.
Proof.
  induction bindings as [ | [id code | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ Hok [= <-]. exact Hok.
  - intros H Hok.
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply IH; [ intros id' code' Hin; apply (H id' code'); right; exact Hin | ].
    intros id' code'. simpl.
    destruct (Pos.eqb id' id);
      [ intros [= <-]; apply (H id code); left; reflexivity | apply Hok ].
  - intros H Hok.
    apply IH; [ intros id' code' Hin; apply (H id' code'); right; exact Hin | ].
    exact Hok.
  - intros H Hok.
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const); simpl; [ | discriminate ].
    apply IH; [ intros id' code' Hin; apply (H id' code'); right; exact Hin | ].
    exact Hok.
Qed.

(* Where a call and the application of a continuation land. *)
Lemma terms_through_over (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_slots_through c f a b d
    (Over_application remaining remaining_arity return_arity return_to
       exn_to)
  <-> cont_slots_through c f a b d return_to
      ∧ cont_slots_through c f a b d exn_to.
Proof. reflexivity. Qed.

Lemma terms_through_code (statics : static_env) (hp : heap)
    (rd : read_record) :
  code_slots_through c f a b d statics ->
  ∀ (called : code_id) (code : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some code ->
    cont_slots_through c f a b d return_to ->
    cont_slots_through c f a b d exn_to ->
    terms_through
      (State (code_body code) callee_vars
         [(code_return_continuation code, return_to);
          (code_exn_continuation code, exn_to)]
         statics hp rd).
Proof.
  intros S called code callee_vars return_to exn_to Hlookup Hreturn Hexn.
  unfold terms_through. simpl_state.
  refine (conj _ (conj (conj Hreturn (conj Hexn I)) S)).
  exists []. split; [ apply made_sight_holds_nil | ].
  pose proof (S called code Hlookup) as H.
  destruct code as [params my_closure ret exn metadata body]. exact H.
Qed.

Lemma terms_through_handler (statics : static_env) (hp : heap)
    (rd : read_record) :
  code_slots_through c f a b d statics ->
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (handler_vars : env),
    cont_slots_through c f a b d
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    bind_params params vs def_vars = Some handler_vars ->
    terms_through
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros S params metadata handler def_vars def_conts recursive vs
    handler_vars Cr Hbind.
  unfold terms_through. simpl_state. cbn [cont_slots_through] in Cr.
  destruct Cr as ((known & Hhold & Hh & Hrec) & Cdef).
  refine (conj _ (conj (conts_slots_through_app _ _ _ Cdef) S)).
  - exists (made_without_all params known). split; [ | exact Hh ].
    exact (made_sight_holds_bind_params params known vs def_vars handler_vars
             Hhold Hbind).
  - exact (rec_closures_slots_through known def_vars def_conts recursive
             recursive Hhold Hrec Cdef Hrec).
Qed.

Lemma terms_through_step (s s' : state) :
  terms_through s -> step s = Step s' -> terms_through s'.
Proof.
  intros ((known & Hhold & P) & C & S) Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind defining body | bindings body | k h body | handlers body | ap
       | ac | scrutinee arms | ];
    autorewrite with step in Hstep.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ defining) as [[[v hp'] rd'] | ] eqn:Hexec;
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. unfold terms_through. simpl_state.
    refine (conj _ (conj C S)).
    exists (made_under_let known x defining). split.
    + exact (made_sight_holds_exec _ _ _ _ _ known vars x defining v Hhold
               Hexec).
    + cbn [sets_through] in P. rewrite sets_through_give_app in P.
      apply andb_true_iff in P as [_ P]. exact P.
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold terms_through. simpl_state.
    cbn [sets_through] in P.
    rewrite sets_through_give_app, bindings_sets_through_give in P.
    apply andb_true_iff in P as [Pbindings Pbody].
    refine (conj _ (conj C _)).
    + exists known. exact (conj Hhold Pbody).
    + apply (eval_static_bindings_code_slots_through vars bindings statics
               statics'); [ | exact S | exact Heval ].
      intros id code Hin. rewrite forallb_forall in Pbindings.
      exact (Pbindings _ Hin).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold terms_through. simpl_state.
    cbn [sets_through] in P. rewrite sets_through_give_app in P.
    apply andb_true_iff in P as [Hh Pbody].
    refine (conj _ (conj (conj _ C) S)).
    + exists known. exact (conj Hhold Pbody).
    + cbn [snd cont_slots_through]. split; [ | exact C ].
      exists known. exact (conj Hhold (conj Hh eq_refl)).
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold terms_through. simpl_state.
    cbn [sets_through] in P.
    rewrite sets_through_give_app, handlers_sets_through_give in P.
    apply andb_true_iff in P as [Phandlers Pbody].
    refine (conj _ (conj (conts_slots_through_app _ _ _ C) S)).
    + exists known. exact (conj Hhold Pbody).
    + exact (rec_closures_slots_through known vars conts handlers handlers
               Hhold Phandlers C Phandlers).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    refine
      (call_function_lands_bound statics _ rd_ (cont_slots_through c f a b d)
         terms_through (fun cv extra H => H) terms_through_over
         (terms_through_code statics _ rd_ S)
         (terms_through_handler statics _ rd_ S) _ _ _ _ _ _ _ s'
         (resolve_slots_through _ conts C) _ Hstep).
    apply with_extra_args_slots_through, resolve_slots_through, C.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    exact
      (apply_cont_value_lands_bound statics _ rd_
         (cont_slots_through c f a b d) terms_through (fun cv extra H => H)
         terms_through_over (terms_through_code statics _ rd_ S)
         (terms_through_handler statics _ rd_ S) _ vs s'
         (resolve_slots_through _ conts C) Hstep).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    injection Hstep as <-. unfold terms_through. simpl_state.
    refine (conj _ (conj C S)).
    exists known. split; [ exact Hhold | reflexivity ].
  - discriminate.
Qed.

Lemma state_slots_through_terms (s : state) :
  state_slots_through c f a b d s
  <-> terms_through s
      ∧ state_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated)
          s.
Proof. unfold state_slots_through, terms_through, Pv, Qv. tauto. Qed.

Lemma state_slots_through_kept :
  kept_by_step (state_slots_through c f a b d).
Proof.
  intros s s' Hinv Hstep.
  apply state_slots_through_terms in Hinv as [Hterms Hvalues].
  apply state_slots_through_terms. split.
  - exact (terms_through_step s s' Hterms Hstep).
  - pose proof (values_through_step s (proj1 Hterms) Hvalues) as H.
    rewrite Hstep in H. exact H.
Qed.

Lemma closure_slot_through_aux (s : state) (x : variable) (v : value)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  state_slots_through c f a b d s ->
  lookup x (state_vars s) = Some v ->
  closure_in_set (state_statics s) v = Some (f, function_decls, slot_values) ->
  lookup f function_decls = Some c ->
  ∃ (inner : value) (inner_slots : list (value_slot * value)) (w : value),
    lookup a slot_values = Some inner
    ∧ closure_value_slots inner = Some inner_slots
    ∧ lookup d inner_slots = Some w
    ∧ lookup b slot_values = Some w.
Proof.
  intros (_ & _ & _ & Hvars & _ & Hstatics & _) Hx Hin_set Hcode.
  destruct (closure_in_set_through (state_statics s) v f function_decls
              slot_values Hstatics
              (env_values_all_lookup _ (state_vars s) x v Hvars Hx) Hin_set)
    as [Hset _].
  apply Hset. unfold has_code_at. rewrite Hcode. apply Pos.eqb_refl.
Qed.

End Slots_through.

Theorem state_slots_through_from : state_slots_through_from_statement.
Proof.
  intros c f a b d e conts statics He Hconts Hvalues Hcode Hstatics.
  unfold state_slots_through, state_values_all. simpl_state.
  refine (conj _ (conj Hconts (conj Hcode _))).
  - exists []. split; [ apply made_sight_holds_nil | exact He ].
  - exact
      (conj (Forall_nil _)
         (conj Hvalues (conj Hstatics (heap_values_all_empty _)))).
Qed.

Theorem state_slots_through_initial : state_slots_through_initial_statement.
Proof.
  intros c f a b d e He.
  apply state_slots_through_from; [ exact He | exact I | exact I | | ].
  - intros id code Hlookup. discriminate Hlookup.
  - apply Forall_nil.
Qed.

Theorem state_slots_through_step : state_slots_through_step_statement.
Proof. intros c f a b d. exact (state_slots_through_kept c f a b d). Qed.

Theorem closure_slot_through : closure_slot_through_statement.
Proof.
  intros c f a b d s x v function_decls slot_values.
  apply closure_slot_through_aux.
Qed.

(* A closure value read as a closure of a set has the value slots that
   it holds. *)
Lemma closure_value_slots_in_set (statics : static_env) (v : value)
    (inner_slots : list (value_slot * value)) (at_slot : function_slot)
    (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  closure_value_slots v = Some inner_slots ->
  closure_in_set statics v = Some (at_slot, decls, slot_values) ->
  slot_values = inner_slots.
Proof.
  destruct v as [k | code slots | at_slot' set_decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    simpl; try discriminate.
  - intros [= <-] [= _ _ <-]. reflexivity.
  - intros [= <-]. destruct (lookup at_slot' set_decls); [ | discriminate ].
    simpl. intros [= _ _ <-]. reflexivity.
Qed.

Theorem projection_through : projection_through_statement.
Proof.
  intros c f g a b d s x v inner w function_decls slot_values Hstate Hx
    Hin_set Hcode Ha Hd.
  destruct (closure_slot_through c f a b d s x v function_decls slot_values
              Hstate Hx Hin_set Hcode)
    as (inner' & inner_slots & w' & Hinner & Hclosure & Hw & Hb).
  unfold project_value_slot in Ha |- *. rewrite Hin_set in Ha |- *.
  simpl in Ha |- *. rewrite Pos.eqb_refl in Ha |- *.
  rewrite Hinner in Ha. injection Ha as <-.
  unfold project_value_slot in Hd.
  destruct (closure_in_set (state_statics s) inner')
    as [[[at_slot decls] slots] | ] eqn:Hinner_set; [ | discriminate ].
  simpl in Hd. destruct (Pos.eqb at_slot g); [ | discriminate ].
  rewrite (closure_value_slots_in_set _ _ _ _ _ _ Hclosure Hinner_set) in Hd.
  rewrite Hw in Hd. injection Hd as <-. exact Hb.
Qed.

(* ======================================================================== *)
(* What the rewrite keeps                                                   *)
(* ======================================================================== *)

(* What is put in the place of a defining expression is the expression,
   or a projection of a value slot in the place of one. *)
Lemma read_through_cases (seen : list set_through) (known : closure_codes)
    (through : projections) (d : named) :
  read_through seen known through d = d
  ∨ ∃ (g f : function_slot) (slot a b : value_slot) (y x : variable)
      (c : code_id),
      d = Project_value_slot_from g slot (Var y)
      ∧ lookup y through = Some (x, f, a)
      ∧ lookup x known = Some c
      ∧ slot_through seen c f a slot = Some b
      ∧ read_through seen known through d
        = Project_value_slot_from f b (Var x).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | g slot closure | move_from move_to closure];
    try (left; reflexivity).
  destruct closure as [y | sym | k]; try (left; reflexivity).
  cbn [read_through].
  destruct (lookup y through) as [[[x f] a] | ] eqn:Hy; [ | left; reflexivity ].
  destruct (lookup x known) as [c | ] eqn:Hx; [ | left; reflexivity ].
  destruct (slot_through seen c f a slot) as [b | ] eqn:Hb;
    [ | left; reflexivity ].
  right. exists g, f, slot, a, b, y, x, c. repeat split; assumption.
Qed.

Lemma read_through_sets_made (seen : list set_through)
    (known : closure_codes) (through : projections) (d : named) :
  named_sets_made (read_through seen known through d) = named_sets_made d.
Proof.
  destruct (read_through_cases seen known through d)
    as [-> | (g & f & slot & a & b & y & x & c & -> & _ & _ & _ & ->)];
    reflexivity.
Qed.

Lemma read_through_sets_through (seen : list set_through)
    (known : closure_codes) (through : projections) (sight : made_sight)
    (d : named) :
  named_sets_through sight (read_through seen known through d)
  = named_sets_through sight d.
Proof.
  destruct (read_through_cases seen known through d)
    as [-> | (g & f & slot & a & b & y & x & c & -> & _ & _ & _ & ->)];
    reflexivity.
Qed.

Lemma read_through_sets_seen (seen : list set_through)
    (known : closure_codes) (through : projections) (sight : closure_sight)
    (d : named) :
  named_sets_seen sight (read_through seen known through d)
  = named_sets_seen sight d.
Proof.
  destruct (read_through_cases seen known through d)
    as [-> | (g & f & slot & a & b & y & x & c & -> & _ & _ & _ & ->)];
    reflexivity.
Qed.

Lemma read_through_made_under_let (seen : list set_through)
    (known : closure_codes) (through : projections) (sight : made_sight)
    (x : variable) (d : named) :
  made_under_let sight x (read_through seen known through d)
  = made_under_let sight x d.
Proof.
  destruct (read_through_cases seen known through d)
    as [-> | (g & f & slot & a & b & y & x' & c & -> & _ & _ & _ & ->)];
    reflexivity.
Qed.

Lemma read_through_sight_under_let (seen : list set_through)
    (known : closure_codes) (through : projections) (sight : closure_sight)
    (x : variable) (d : named) :
  sight_under_let sight x (read_through seen known through d)
  = sight_under_let sight x d.
Proof.
  destruct (read_through_cases seen known through d)
    as [-> | (g & f & slot & a & b & y & x' & c & -> & _ & _ & _ & ->)];
    reflexivity.
Qed.

(* It makes no set of closures and changes none, as they are made and
   as they are seen, here and by FlambdaSlotsOfCodeSpec.sets_seen. *)
Fixpoint know_through_in_body_sets_made (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  ∀ (known : closure_codes) (through : projections),
    sets_made (know_through_in_body sets codes seen known through e)
    = sets_made e
with know_through_in_handler_sets_made (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  ∀ (known : closure_codes) (through : projections),
    handler_sets_made (know_through_in_handler sets codes seen known through h)
    = handler_sets_made h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known through; cbn [know_through_in_body sets_made];
      try reflexivity.
    + rewrite (know_through_in_body_sets_made sets codes seen body),
        read_through_sets_made.
      reflexivity.
    + rewrite (know_through_in_body_sets_made sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_handler_sets_made sets codes seen h),
        (know_through_in_body_sets_made sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_body_sets_made sets codes seen body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (know_through_in_handler_sets_made sets codes seen h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known through.
    cbn [know_through_in_handler handler_sets_made].
    apply know_through_in_body_sets_made.
Qed.

Fixpoint know_through_in_body_sets_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  ∀ (known : closure_codes) (through : projections) (sight : made_sight),
    sets_through sight (know_through_in_body sets codes seen known through e)
    = sets_through sight e
with know_through_in_handler_sets_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  ∀ (known : closure_codes) (through : projections) (sight : made_sight),
    handler_sets_through sight
      (know_through_in_handler sets codes seen known through h)
    = handler_sets_through sight h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known through sight; cbn [know_through_in_body sets_through];
      try reflexivity.
    + rewrite read_through_sets_through, read_through_made_under_let,
        (know_through_in_body_sets_through sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_body_sets_through sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_handler_sets_through sets codes seen h),
        (know_through_in_body_sets_through sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_body_sets_through sets codes seen body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (know_through_in_handler_sets_through sets codes seen h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known through sight.
    cbn [know_through_in_handler handler_sets_through].
    apply know_through_in_body_sets_through.
Qed.

Fixpoint know_through_in_body_sets_seen (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  ∀ (known : closure_codes) (through : projections) (sight : closure_sight),
    sets_seen sight (know_through_in_body sets codes seen known through e)
    = sets_seen sight e
with know_through_in_handler_sets_seen (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  ∀ (known : closure_codes) (through : projections) (sight : closure_sight),
    handler_sets_seen sight
      (know_through_in_handler sets codes seen known through h)
    = handler_sets_seen sight h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known through sight; cbn [know_through_in_body sets_seen];
      try reflexivity.
    + rewrite read_through_sets_seen, read_through_sight_under_let,
        (know_through_in_body_sets_seen sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_body_sets_seen sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_handler_sets_seen sets codes seen h),
        (know_through_in_body_sets_seen sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_body_sets_seen sets codes seen body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (know_through_in_handler_sets_seen sets codes seen h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known through sight.
    cbn [know_through_in_handler handler_sets_seen].
    apply know_through_in_body_sets_seen.
Qed.

Fixpoint know_through_in_body_groups_whole (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  ∀ (known : closure_codes) (through : projections),
    groups_whole (know_through_in_body sets codes seen known through e)
    = groups_whole e
with know_through_in_handler_groups_whole (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  ∀ (known : closure_codes) (through : projections),
    handler_groups_whole
      (know_through_in_handler sets codes seen known through h)
    = handler_groups_whole h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known through; cbn [know_through_in_body groups_whole];
      try reflexivity.
    + apply know_through_in_body_groups_whole.
    + rewrite (know_through_in_body_groups_whole sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_handler_groups_whole sets codes seen h),
        (know_through_in_body_groups_whole sets codes seen body).
      reflexivity.
    + rewrite (know_through_in_body_groups_whole sets codes seen body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (know_through_in_handler_groups_whole sets codes seen h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known through.
    cbn [know_through_in_handler handler_groups_whole].
    apply know_through_in_body_groups_whole.
Qed.

Fixpoint know_through_in_body_binds_statics (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) :
  ∀ (known : closure_codes) (through : projections),
    binds_statics (know_through_in_body sets codes seen known through e)
    = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros known through; simpl; try reflexivity.
  - apply know_through_in_body_binds_statics.
  - rewrite know_through_in_body_binds_statics. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite !know_through_in_body_binds_statics. reflexivity.
  - rewrite know_through_in_body_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite know_through_in_body_binds_statics, IH. reflexivity.
Qed.

(* The same of the rewrite of the code of a unit. *)
Lemma know_through_in_code_sets_made (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (id : code_id) (c : code) :
  code_sets_made (know_through_in_code sets codes seen id c) = code_sets_made c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [know_through_in_code].
  destruct (cm_is_stub metadata || among my_closure params);
    [ reflexivity | ].
  cbn [code_sets_made]. apply know_through_in_body_sets_made.
Qed.

Lemma know_through_in_bindings_sets_made (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (bindings : list static_binding) :
  flat_map binding_sets_made
    (map (know_through_in_binding sets codes seen) bindings)
  = flat_map binding_sets_made bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    cbn [map flat_map know_through_in_binding binding_sets_made];
    [ reflexivity | | | ]; rewrite IH; [ | reflexivity | reflexivity ].
  rewrite know_through_in_code_sets_made. reflexivity.
Qed.

Fixpoint know_code_through_sets_made (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  sets_made (know_code_through sets codes seen e) = sets_made e
with know_code_through_in_handler_sets_made (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  handler_sets_made (know_code_through_in_handler sets codes seen h)
  = handler_sets_made h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [know_code_through sets_made]; try reflexivity.
    + rewrite (know_code_through_sets_made sets codes seen body). reflexivity.
    + rewrite (know_code_through_sets_made sets codes seen body),
        know_through_in_bindings_sets_made.
      reflexivity.
    + rewrite (know_code_through_in_handler_sets_made sets codes seen h),
        (know_code_through_sets_made sets codes seen body).
      reflexivity.
    + rewrite (know_code_through_sets_made sets codes seen body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (know_code_through_in_handler_sets_made sets codes seen h), IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [know_code_through_in_handler handler_sets_made].
    apply know_code_through_sets_made.
Qed.

Lemma know_through_in_code_sets_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (id : code_id) (c : code) :
  code_sets_through (know_through_in_code sets codes seen id c)
  = code_sets_through c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [know_through_in_code].
  destruct (cm_is_stub metadata || among my_closure params);
    [ reflexivity | ].
  cbn [code_sets_through]. apply know_through_in_body_sets_through.
Qed.

Lemma know_through_in_code_sets_seen (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (id : code_id) (c : code) :
  code_sets_seen (know_through_in_code sets codes seen id c)
  = code_sets_seen c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [know_through_in_code].
  destruct (cm_is_stub metadata || among my_closure params);
    [ reflexivity | ].
  cbn [code_sets_seen]. apply know_through_in_body_sets_seen.
Qed.

Lemma know_through_in_bindings_sets_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (sight : made_sight)
    (bindings : list static_binding) :
  flat_map (binding_sets_through sight)
    (map (know_through_in_binding sets codes seen) bindings)
  = flat_map (binding_sets_through sight) bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    cbn [map flat_map know_through_in_binding binding_sets_through];
    [ reflexivity | | | ]; rewrite IH; [ | reflexivity | reflexivity ].
  rewrite know_through_in_code_sets_through. reflexivity.
Qed.

Lemma know_through_in_bindings_sets_seen (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (sight : closure_sight)
    (bindings : list static_binding) :
  flat_map (binding_sets_seen sight)
    (map (know_through_in_binding sets codes seen) bindings)
  = flat_map (binding_sets_seen sight) bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    cbn [map flat_map know_through_in_binding binding_sets_seen];
    [ reflexivity | | | ]; rewrite IH; [ | reflexivity | reflexivity ].
  rewrite know_through_in_code_sets_seen. reflexivity.
Qed.

Fixpoint know_code_through_sets_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  ∀ sight : made_sight,
    sets_through sight (know_code_through sets codes seen e)
    = sets_through sight e
with know_code_through_in_handler_sets_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  ∀ sight : made_sight,
    handler_sets_through sight (know_code_through_in_handler sets codes seen h)
    = handler_sets_through sight h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros sight; cbn [know_code_through sets_through]; try reflexivity.
    + rewrite (know_code_through_sets_through sets codes seen body).
      reflexivity.
    + rewrite (know_code_through_sets_through sets codes seen body),
        know_through_in_bindings_sets_through.
      reflexivity.
    + rewrite (know_code_through_in_handler_sets_through sets codes seen h),
        (know_code_through_sets_through sets codes seen body).
      reflexivity.
    + rewrite (know_code_through_sets_through sets codes seen body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (know_code_through_in_handler_sets_through sets codes seen h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros sight.
    cbn [know_code_through_in_handler handler_sets_through].
    apply know_code_through_sets_through.
Qed.

Fixpoint know_code_through_sets_seen (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  ∀ sight : closure_sight,
    sets_seen sight (know_code_through sets codes seen e) = sets_seen sight e
with know_code_through_in_handler_sets_seen (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  ∀ sight : closure_sight,
    handler_sets_seen sight (know_code_through_in_handler sets codes seen h)
    = handler_sets_seen sight h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros sight; cbn [know_code_through sets_seen]; try reflexivity.
    + rewrite (know_code_through_sets_seen sets codes seen body). reflexivity.
    + rewrite (know_code_through_sets_seen sets codes seen body),
        know_through_in_bindings_sets_seen.
      reflexivity.
    + rewrite (know_code_through_in_handler_sets_seen sets codes seen h),
        (know_code_through_sets_seen sets codes seen body).
      reflexivity.
    + rewrite (know_code_through_sets_seen sets codes seen body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (know_code_through_in_handler_sets_seen sets codes seen h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros sight.
    cbn [know_code_through_in_handler handler_sets_seen].
    apply know_code_through_sets_seen.
Qed.

Lemma know_through_in_code_groups_whole (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (id : code_id) (c : code) :
  code_groups_whole (know_through_in_code sets codes seen id c)
  = code_groups_whole c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [know_through_in_code].
  destruct (cm_is_stub metadata || among my_closure params);
    [ reflexivity | ].
  cbn [code_groups_whole]. apply know_through_in_body_groups_whole.
Qed.

Lemma know_through_in_bindings_groups_whole (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (bindings : list static_binding) :
  forallb binding_groups_whole
    (map (know_through_in_binding sets codes seen) bindings)
  = forallb binding_groups_whole bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    cbn [map forallb know_through_in_binding binding_groups_whole];
    [ reflexivity | | | ]; rewrite IH; [ | reflexivity | reflexivity ].
  rewrite know_through_in_code_groups_whole. reflexivity.
Qed.

Lemma binds_symbol_to_know_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (s : symbol) (const : static_const)
    (binding : static_binding) :
  binds_symbol_to s const (know_through_in_binding sets codes seen binding)
  = binds_symbol_to s const binding.
Proof. destruct binding; reflexivity. Qed.

Lemma existsb_binds_symbol_to_know_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (s : symbol) (const : static_const)
    (bindings : list static_binding) :
  existsb (binds_symbol_to s const)
    (map (know_through_in_binding sets codes seen) bindings)
  = existsb (binds_symbol_to s const) bindings.
Proof.
  induction bindings as [ | binding bindings IH]; cbn [map existsb];
    [ reflexivity | ].
  rewrite binds_symbol_to_know_through, IH. reflexivity.
Qed.

Lemma static_set_whole_know_through (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (bindings : list static_binding)
    (binding : static_binding) :
  static_set_whole (map (know_through_in_binding sets codes seen) bindings)
    (know_through_in_binding sets codes seen binding)
  = static_set_whole bindings binding.
Proof.
  destruct binding as [id c | id | sym const];
    cbn [know_through_in_binding static_set_whole]; try reflexivity.
  destruct const
    as [code slots | slot closure_symbols decls slots | tag fields
       | contents | contents | elements];
    try reflexivity.
  cbn [static_set_whole]. f_equal. apply forallb_same. intros [slot' s'].
  apply existsb_binds_symbol_to_know_through.
Qed.

Lemma know_through_in_bindings_whole (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (bindings : list static_binding) :
  forallb
    (static_set_whole (map (know_through_in_binding sets codes seen) bindings))
    (map (know_through_in_binding sets codes seen) bindings)
  = forallb (static_set_whole bindings) bindings.
Proof.
  generalize bindings at 1 3. intros all.
  induction bindings as [ | binding bindings IH]; cbn [map forallb];
    [ reflexivity | ].
  rewrite static_set_whole_know_through, IH. reflexivity.
Qed.

Fixpoint know_code_through_groups_whole (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  groups_whole (know_code_through sets codes seen e) = groups_whole e
with know_code_through_in_handler_groups_whole (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  handler_groups_whole (know_code_through_in_handler sets codes seen h)
  = handler_groups_whole h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [know_code_through groups_whole]; try reflexivity.
    + apply know_code_through_groups_whole.
    + rewrite (know_code_through_groups_whole sets codes seen body),
        know_through_in_bindings_whole, know_through_in_bindings_groups_whole.
      reflexivity.
    + rewrite (know_code_through_in_handler_groups_whole sets codes seen h),
        (know_code_through_groups_whole sets codes seen body).
      reflexivity.
    + rewrite (know_code_through_groups_whole sets codes seen body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (know_code_through_in_handler_groups_whole sets codes seen h), IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [know_code_through_in_handler handler_groups_whole].
    apply know_code_through_groups_whole.
Qed.

Fixpoint know_code_through_binds_statics (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) :
  binds_statics (know_code_through sets codes seen e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    cbn [know_code_through binds_statics]; try reflexivity.
  - apply know_code_through_binds_statics.
  - rewrite (know_code_through_binds_statics sets codes seen body). f_equal.
    induction bindings as [ | [id c | id | sym const] bindings IH];
      cbn [map existsb know_through_in_binding binding_binds]; congruence.
  - destruct h as [params metadata handler].
    cbn [know_code_through_in_handler handler_binds_statics].
    rewrite !know_code_through_binds_statics. reflexivity.
  - rewrite (know_code_through_binds_statics sets codes seen body). f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      cbn [map existsb know_code_through_in_handler handler_binds_statics];
      [ reflexivity | ].
    rewrite know_code_through_binds_statics, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* The rewrite: the invariant                                               *)
(* ======================================================================== *)

Lemma closure_in_set_extended (statics statics' : static_env) (v : value)
    (closure :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value)) :
  statics_extend statics statics' ->
  closure_in_set statics v = Some closure ->
  closure_in_set statics' v = Some closure.
Proof.
  intros [_ Hvalues].
  destruct v as [k | code slots | at_slot decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    simpl; try exact (fun H => H).
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
    [ | discriminate ].
  rewrite (Hvalues _ _ Hs). exact (fun H => H).
Qed.

Lemma project_value_slot_extended (statics statics' : static_env)
    (f : function_slot) (a : value_slot) (v w : value) :
  statics_extend statics statics' ->
  project_value_slot statics f a v = Some w ->
  project_value_slot statics' f a v = Some w.
Proof.
  intros Hextend. unfold project_value_slot.
  destruct (closure_in_set statics v) as [closure | ] eqn:Hin_set;
    [ | discriminate ].
  rewrite (closure_in_set_extended statics statics' v closure Hextend
             Hin_set).
  exact (fun H => H).
Qed.

(* What is remembered holds: every projection remembered, an entry that
   another hides included, is the value of its variable. *)
Definition paths_hold (through : projections) (vars : env)
    (statics : static_env) : Prop :=
  ∀ (y x : variable) (f : function_slot) (a : value_slot),
    (y, (x, f, a)) ∈ through ->
    ∃ v inner : value,
      lookup x vars = Some v
      ∧ project_value_slot statics f a v = Some inner
      ∧ lookup y vars = Some inner.

Lemma paths_hold_nil (vars : env) (statics : static_env) :
  paths_hold [] vars statics.
Proof. intros y x f a []. Qed.

Lemma in_through_without (x : variable) (through : projections)
    (y x0 : variable) (f : function_slot) (a : value_slot) :
  (y, (x0, f, a)) ∈ through_without x through ->
  (y, (x0, f, a)) ∈ through ∧ Pos.eqb y x = false ∧ Pos.eqb x0 x = false.
Proof.
  unfold through_without, unbind. intros Hin.
  apply filter_In in Hin as [Hin Hbase]. apply filter_In in Hin as [Hin Hkey].
  cbn [fst snd] in Hbase, Hkey.
  apply negb_true_iff in Hbase. apply negb_true_iff in Hkey.
  rewrite Pos.eqb_sym in Hbase. rewrite Pos.eqb_sym in Hkey.
  repeat split; assumption.
Qed.

Lemma paths_hold_without (through : projections) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  paths_hold through vars statics ->
  paths_hold (through_without x through) ((x, v) :: vars) statics.
Proof.
  intros Hhold y x0 f a Hin.
  destruct (in_through_without x through y x0 f a Hin) as (Hin' & Hy & Hx0).
  destruct (Hhold y x0 f a Hin') as (v0 & inner & Hv0 & Hproject & Hinner).
  exists v0, inner. cbn [lookup]. rewrite Hy, Hx0.
  repeat split; assumption.
Qed.

Lemma paths_hold_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (through : projections) (vs : list value) (vars handler_vars : env),
    paths_hold through vars statics ->
    bind_params params vs vars = Some handler_vars ->
    paths_hold (through_without_all params through) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros through [ | v vs] vars handler_vars Hhold; simpl;
    try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, paths_hold_without, Hhold.
Qed.

Lemma paths_hold_extend (through : projections) (vars : env)
    (statics statics' : static_env) :
  statics_extend statics statics' ->
  paths_hold through vars statics ->
  paths_hold through vars statics'.
Proof.
  intros Hextend Hhold y x f a Hin.
  destruct (Hhold y x f a Hin) as (v & inner & Hv & Hproject & Hinner).
  exists v, inner. split; [ exact Hv | split; [ | exact Hinner ] ].
  exact (project_value_slot_extended statics statics' f a v inner Hextend
           Hproject).
Qed.

Lemma paths_hold_under_let (through : projections) (x : variable)
    (d : named) (vars : env) (statics : static_env) (v : value) :
  paths_hold through vars statics ->
  eval_named statics vars d = Some v ->
  paths_hold (through_under_let through x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd.
  pose proof (paths_hold_without through vars statics x v Hhold) as Hforgot.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | f a closure | move_from move_to closure];
    try exact Hforgot.
  destruct closure as [from | sym | k]; try exact Hforgot.
  cbn [through_under_let].
  destruct (Pos.eqb from x) eqn:Hfrom; [ exact Hforgot | ].
  intros y x0 f0 a0 [Heq | Hin]; [ | exact (Hforgot y x0 f0 a0 Hin) ].
  injection Heq as <- <- <- <-.
  cbn [eval_named eval_simple] in Hd.
  destruct (lookup from vars) as [v0 | ] eqn:Hv0; [ | discriminate ].
  simpl in Hd.
  exists v0, v. cbn [lookup]. rewrite Hfrom, Pos.eqb_refl.
  repeat split; assumption.
Qed.

(* The same where the named expression is run on a heap. *)
Lemma paths_hold_under_let_exec (through : projections) (x : variable)
    (d : named) (vars : env) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (v : value) :
  paths_hold through vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  paths_hold (through_under_let through x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | f a closure | move_from move_to closure];
    try (pose proof (fun H => exec_named_pure _ _ _ _ _ _ _ _ H Hd) as Hpure;
         destruct (Hpure eq_refl) as (Hw & _ & _);
         exact (paths_hold_under_let through x _ vars statics v Hhold Hw)).
  exact (paths_hold_without through vars statics x v Hhold).
Qed.

Section Through_known.

Variable sets : list set_made.
Variable codes : list set_seen.
Variable seen : list set_through.

(* Every fact of a value slot read through another closure that the
   sets given pass the test of, as they are seen, the state has. *)
Definition through_facts_hold (s : state) : Prop :=
  ∀ (c : code_id) (f : function_slot) (a b d : value_slot),
    sets_through_give c f a b d seen = true ->
    state_slots_through c f a b d s.

(* The invariant: that of the second pass of
   FlambdaClosureOfCodeSpec.know_own_closures
   (FlambdaClosureOfCodeProofs.slots_inv: what the sets say of the code
   of the closures, and of the code of the closures that their value
   slots hold, the sets [codes] being as
   FlambdaSlotsOfCodeSpec.sets_seen sees them), which is what a
   variable known by a value slot rests on; and the facts of this
   file. *)
Definition through_inv (s : state) : Prop :=
  slots_inv sets codes s ∧ through_facts_hold s.

Lemma through_inv_kept : kept_by_step through_inv.
Proof.
  intros s s' [Hcode Hslots] Hstep. split.
  - exact (slots_inv_kept sets codes s s' Hcode Hstep).
  - intros c f a b d Htest.
    exact (state_slots_through_step c f a b d s s' (Hslots c f a b d Htest)
             Hstep).
Qed.

(* The invariant looks at the expression of a state only to ask the
   tests of it. *)
Lemma through_inv_expr (e e' : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  sets_made e' = sets_made e ->
  groups_whole e' = groups_whole e ->
  (∀ sight : made_sight, sets_through sight e' = sets_through sight e) ->
  (∀ sight : closure_sight, sets_seen sight e' = sets_seen sight e) ->
  through_inv (State e vars conts statics hp rd)
  <-> through_inv (State e' vars conts statics hp rd).
Proof.
  intros Hsets Hwhole Hseen Hcodes.
  pose proof (slots_inv_expr sets codes e e' vars conts statics hp rd Hsets
                Hwhole Hcodes)
    as Hiff.
  unfold through_inv, through_facts_hold, state_slots_through,
    expr_slots_through, state_values_all.
  simpl_state.
  split; intros [H1 H2]; (split; [ apply Hiff, H1 | ]);
    intros c f a b d Htest;
    destruct (H2 c f a b d Htest) as ((known & Hhold & P) & Hrest);
    (split; [ exists known; split; [ exact Hhold | ] | exact Hrest ]).
  - rewrite Hseen. exact P.
  - rewrite <- Hseen. exact P.
Qed.

(* What the invariant gives of the closures that the variables are
   bound to. *)
Definition through_facts_read (vars : env) (statics : static_env) : Prop :=
  ∀ (c : code_id) (f g : function_slot) (a b d : value_slot) (x : variable)
    (v inner w : value) (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)),
    sets_through_give c f a b d seen = true ->
    lookup x vars = Some v ->
    closure_in_set statics v = Some (f, decls, slot_values) ->
    lookup f decls = Some c ->
    project_value_slot statics f a v = Some inner ->
    project_value_slot statics g d inner = Some w ->
    project_value_slot statics f b v = Some w.

Lemma through_inv_facts_read (s : state) :
  through_inv s -> through_facts_read (state_vars s) (state_statics s).
Proof.
  intros [_ Hslots] c f g a b d x v inner w decls slot_values Htest Hx
    Hin_set Hcode Ha Hd.
  exact (projection_through c f g a b d s x v inner w decls slot_values
           (Hslots c f a b d Htest) Hx Hin_set Hcode Ha Hd).
Qed.

(* The test that [slot_through] has passed. *)
Lemma slot_through_test (c : code_id) (f : function_slot)
    (a d b : value_slot) :
  slot_through seen c f a d = Some b ->
  sets_through_give c f a b d seen = true.
Proof.
  unfold slot_through.
  destruct (find (fun set => has_code_at c f (fst set)) seen) as [set | ];
    [ | discriminate ].
  destruct
    (find (fun entry => sets_through_give c f a (fst entry) d seen)
       (snd set))
    as [entry | ] eqn:Hfind;
    [ | discriminate ].
  simpl. intros [= <-]. exact (proj2 (find_some _ _ Hfind)).
Qed.

(* What is put in the place of a projection has its value. *)
Lemma read_through_sound (known : closure_codes) (through : projections)
    (vars : env) (statics : static_env) (d : named) (v : value) :
  through_facts_read vars statics ->
  codes_hold known vars statics ->
  paths_hold through vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (read_through seen known through d) = Some v.
Proof.
  intros Hread Hhold Hpaths.
  destruct (read_through_cases seen known through d)
    as [-> | (g & f & slot & a & b & y & x & c & -> & Hy & Hx & Hb & ->)];
    [ exact (fun H => H) | ].
  destruct (Hhold x c Hx) as (vx & slots & Hvx & Hclosure).
  destruct (Hpaths y x f a (lookup_in _ _ _ Hy))
    as (vx' & inner & Hvx' & Ha & Hinner).
  rewrite Hvx in Hvx'. injection Hvx' as <-.
  cbn [eval_named eval_simple]. rewrite Hinner, Hvx. simpl. intros Hd.
  pose proof Ha as Ha'. unfold project_value_slot in Ha'.
  destruct (closure_in_set statics vx) as [[[at_slot decls] slot_values] | ]
    eqn:Hin_set;
    [ | discriminate ].
  simpl in Ha'.
  destruct (Pos.eqb_spec at_slot f) as [-> | _]; [ | discriminate ].
  pose proof (closure_of_in_set statics vx c slots f decls slot_values
                Hclosure Hin_set) as Hcode.
  exact (Hread c f g a b slot x vx inner v decls slot_values
           (slot_through_test c f a slot b Hb) Hvx Hin_set Hcode Ha Hd).
Qed.

(* The same where the named expression is run on a heap. *)
Lemma read_through_sound_exec (known : closure_codes)
    (through : projections) (vars : env) (statics : static_env)
    (hp hp' : heap) (rd rd' : read_record) (d : named) (v : value) :
  through_facts_read vars statics ->
  codes_hold known vars statics ->
  paths_hold through vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars (read_through seen known through d)
  = Some (v, hp', rd').
Proof.
  intros Hread Hhold Hpaths Hd.
  destruct (read_through_cases seen known through d)
    as [-> | (g & f & slot & a & b & y & x & c & Heq & Hy & Hx & Hb & Hto)];
    [ exact Hd | ].
  subst d.
  pose proof (fun H => exec_named_pure _ _ _ _ _ _ _ _ H Hd) as Hpure.
  destruct (Hpure eq_refl) as (Hw & -> & ->).
  pose proof (read_through_sound known through vars statics _ v Hread Hhold
                Hpaths Hw) as Hw'.
  rewrite Hto in Hw' |- *.
  rewrite exec_named_not_prim by (intros ? ?; discriminate).
  rewrite Hw'. reflexivity.
Qed.

(* ======================================================================== *)
(* The rewrite: the body of code                                            *)
(* ======================================================================== *)

Definition through_inv_at (statics0 : static_env) (s : state) : Prop :=
  through_inv s ∧ statics_extend statics0 (state_statics s).

Lemma through_inv_at_kept (statics0 : static_env) :
  kept_by_step (through_inv_at statics0).
Proof.
  intros s s' [Hinv Hextend] Hstep.
  split; [ exact (through_inv_kept s s' Hinv Hstep) | ].
  exact (statics_extend_trans _ _ _ Hextend (step_extends_statics s s' Hstep)).
Qed.

Lemma through_inv_at_expr (statics0 : static_env) (e e' : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  sets_made e' = sets_made e ->
  groups_whole e' = groups_whole e ->
  (∀ sight : made_sight, sets_through sight e' = sets_through sight e) ->
  (∀ sight : closure_sight, sets_seen sight e' = sets_seen sight e) ->
  through_inv_at statics0 (State e vars conts statics hp rd)
  <-> through_inv_at statics0 (State e' vars conts statics hp rd).
Proof.
  intros Hsets Hwhole Hseen Hcodes.
  pose proof (through_inv_expr e e' vars conts statics hp rd Hsets Hwhole
                Hseen Hcodes)
    as Hiff.
  unfold through_inv_at. simpl_state.
  split; intros [H1 H2]; (split; [ apply Hiff, H1 | exact H2 ]).
Qed.

Fixpoint know_through_in_body_refines (e : expr) {struct e} :
  ∀ (known : closure_codes) (through : projections) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    through_inv (State e vars conts statics hp rd) ->
    codes_hold known vars statics ->
    paths_hold through vars statics ->
    refines (State e vars conts statics hp rd)
      (State (know_through_in_body sets codes seen known through e) vars conts
         statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros known through vars conts statics hp rd Hinv Hhold Hpaths;
    cbn [know_through_in_body]; try apply refines_refl.
  - (* Let *)
    pose proof (code_inv_facts_read sets _ (proj1 (proj1 Hinv))) as Hread.
    pose proof (slots_inv_facts_read sets codes _ (proj1 Hinv)) as Hslots.
    pose proof (through_inv_facts_read _ Hinv) as Hthrough. simpl_state.
    apply let_refines_step_forward.
    + intros v hp' rd' Hd.
      exact (read_through_sound_exec known through vars statics hp hp' rd
               rd' d v Hthrough Hhold Hpaths Hd).
    + intros v hp' rd' Hd Hkind. apply know_through_in_body_refines.
      * apply (through_inv_kept _ _ Hinv).
        rewrite step_let_exec, Hd, Hkind, kind_eqb_refl. reflexivity.
      * exact (codes_hold_under_read_exec sets codes known x d vars statics hp
                 hp' rd rd' v Hread Hslots Hhold Hd).
      * exact (paths_hold_under_let_exec through x d vars statics hp hp' rd
                 rd' v Hpaths Hd).
  - (* Let_static *)
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Heval;
      [ | exact I ].
    apply know_through_in_body_refines.
    + apply (through_inv_kept _ _ Hinv).
      rewrite step_let_static, Heval. reflexivity.
    + exact (codes_hold_extend known vars statics extended
               (eval_static_bindings_extend vars bindings statics extended
                  Heval)
               Hhold).
    + exact (paths_hold_extend through vars statics extended
               (eval_static_bindings_extend vars bindings statics extended
                  Heval)
               Hpaths).
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler]. cbn [know_through_in_handler].
    set (handler' :=
           know_through_in_body sets codes seen (forget_code_vars params known)
             (through_without_all params through) handler).
    assert (Hinv' :
      through_inv
        (State
           (Let_cont_non_recursive k (Handler params metadata handler') body)
           vars conts statics hp rd)).
    { revert Hinv. apply through_inv_expr.
      - cbn [sets_made handler_sets_made]. unfold handler'.
        rewrite know_through_in_body_sets_made. reflexivity.
      - cbn [groups_whole handler_groups_whole]. unfold handler'.
        rewrite know_through_in_body_groups_whole. reflexivity.
      - intros sight. cbn [sets_through handler_sets_through]. unfold handler'.
        rewrite know_through_in_body_sets_through. reflexivity.
      - intros sight. cbn [sets_seen handler_sets_seen]. unfold handler'.
        rewrite know_through_in_body_sets_seen. reflexivity. }
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k (Handler params metadata handler') body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_from (through_inv_at statics) k params
               metadata handler handler' body vars conts statics hp rd
               (through_inv_at_kept statics)).
      * exact (conj Hinv (statics_extend_refl statics)).
      * exact (conj Hinv' (statics_extend_refl statics)).
      * intros vs handler_vars conts' statics' hp' rd' _ Hbind. split.
        -- apply through_inv_at_expr; unfold handler';
             [ apply know_through_in_body_sets_made
             | apply know_through_in_body_groups_whole
             | intros sight; apply know_through_in_body_sets_through
             | intros sight; apply know_through_in_body_sets_seen ].
        -- intros [Hentry Hextend]. simpl_state.
           apply know_through_in_body_refines; [ exact Hentry | | ].
           ++ exact (codes_hold_bind_params statics' params known vs vars
                       handler_vars
                       (codes_hold_extend known vars statics statics'
                          Hextend Hhold)
                       Hbind).
           ++ exact (paths_hold_bind_params statics' params through vs vars
                       handler_vars
                       (paths_hold_extend through vars statics statics'
                          Hextend Hpaths)
                       Hbind).
    + apply let_cont_non_recursive_refines_step.
      apply know_through_in_body_refines; [ | exact Hhold | exact Hpaths ].
      apply (through_inv_kept _ _ Hinv'). apply step_let_cont_non_recursive.
  - (* Let_cont_recursive *)
    set (know :=
           fun kh : continuation * continuation_handler =>
             let '(k, h) := kh in
             (k, know_through_in_handler sets codes seen known through h)).
    apply (refines_trans _
      (State (Let_cont_recursive (map know handlers) body) vars conts
         statics hp rd)).
    + assert (Hhandlers :
        ∀ done : list (continuation * continuation_handler),
          through_inv
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp rd)
          ->
          refines
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp rd)
            (State (Let_cont_recursive (done ++ map know handlers) body)
               vars conts statics hp rd)).
      { clear Hinv.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          intros done Hdone; cbn [map]; [ apply refines_refl | ].
        unfold know at 1. cbn [know_through_in_handler].
        set (handler' :=
               know_through_in_body sets codes seen
                 (forget_code_vars params known)
                 (through_without_all params through) handler).
        assert (Hdone' :
          through_inv
            (State
               (Let_cont_recursive
                  (done ++ (k, Handler params metadata handler') :: handlers)
                  body)
               vars conts statics hp rd)).
        { revert Hdone. apply through_inv_expr.
          - cbn [sets_made]. rewrite !flat_map_app.
            cbn [flat_map handler_sets_made]. unfold handler'.
            rewrite know_through_in_body_sets_made. reflexivity.
          - cbn [groups_whole]. rewrite !forallb_app.
            cbn [forallb handler_groups_whole]. unfold handler'.
            rewrite know_through_in_body_groups_whole. reflexivity.
          - intros sight. cbn [sets_through]. rewrite !flat_map_app.
            cbn [flat_map handler_sets_through]. unfold handler'.
            rewrite know_through_in_body_sets_through. reflexivity.
          - intros sight. cbn [sets_seen]. rewrite !flat_map_app.
            cbn [flat_map handler_sets_seen]. unfold handler'.
            rewrite know_through_in_body_sets_seen. reflexivity. }
        apply (refines_trans _
          (State
             (Let_cont_recursive
                (done ++ (k, Handler params metadata handler') :: handlers)
                body)
             vars conts statics hp rd)).
        - apply (let_cont_recursive_handler_refines_from
                   (through_inv_at statics) done k params metadata handler
                   handler' handlers body vars conts statics hp rd
                   (through_inv_at_kept statics)).
          + exact (conj Hdone (statics_extend_refl statics)).
          + exact (conj Hdone' (statics_extend_refl statics)).
          + symmetry. unfold handler'. apply know_through_in_body_binds_statics.
          + intros vs handler_vars conts' statics' hp' rd' _ Hbind. split.
            * apply through_inv_at_expr; unfold handler';
                [ apply know_through_in_body_sets_made
                | apply know_through_in_body_groups_whole
                | intros sight; apply know_through_in_body_sets_through
                | intros sight; apply know_through_in_body_sets_seen ].
            * intros [Hentry Hextend]. simpl_state.
              apply know_through_in_body_refines; [ exact Hentry | | ].
              -- exact (codes_hold_bind_params statics' params known vs vars
                          handler_vars
                          (codes_hold_extend known vars statics statics'
                             Hextend Hhold)
                          Hbind).
              -- exact (paths_hold_bind_params statics' params through vs
                          vars handler_vars
                          (paths_hold_extend through vars statics statics'
                             Hextend Hpaths)
                          Hbind).
        - specialize
            (IH (done ++ [(k, Handler params metadata handler')])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone'). }
      exact (Hhandlers [] Hinv).
    + assert (Hinv' :
        through_inv
          (State (Let_cont_recursive (map know handlers) body) vars conts
             statics hp rd)).
      { revert Hinv. apply through_inv_expr.
        - cbn [sets_made]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
            [ reflexivity | ].
          unfold know at 1. rewrite know_through_in_handler_sets_made, IH.
          reflexivity.
        - cbn [groups_whole]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
            [ reflexivity | ].
          unfold know at 1. rewrite know_through_in_handler_groups_whole, IH.
          reflexivity.
        - intros sight. cbn [sets_through]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
            [ reflexivity | ].
          unfold know at 1. rewrite know_through_in_handler_sets_through, IH.
          reflexivity.
        - intros sight. cbn [sets_seen]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
            [ reflexivity | ].
          unfold know at 1. rewrite know_through_in_handler_sets_seen, IH.
          reflexivity. }
      apply let_cont_recursive_refines_step; [ reflexivity | ].
      intros Hbinds.
      apply know_through_in_body_refines; [ | exact Hhold | exact Hpaths ].
      apply (through_inv_kept _ _ Hinv').
      rewrite step_let_cont_recursive, Hbinds. reflexivity.
Qed.

(* ======================================================================== *)
(* The rewrite: the code of a unit                                          *)
(* ======================================================================== *)

Fixpoint know_code_through_refines_aux (e : expr) {struct e} :
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    through_inv (State e vars conts statics hp rd) ->
    refines (State e vars conts statics hp rd)
      (State (know_code_through sets codes seen e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros vars conts statics hp rd Hinv; cbn [know_code_through];
    try apply refines_refl.
  - (* Let *)
    apply let_refines_step_exec; [ reflexivity | ].
    intros v hp' rd' Hv Hkind. apply know_code_through_refines_aux.
    apply (through_inv_kept _ _ Hinv).
    rewrite step_let_exec, Hv, Hkind, kind_eqb_refl. reflexivity.
  - (* Let_static *)
    apply (refines_trans _
      (State
         (Let_static (map (know_through_in_binding sets codes seen) bindings)
            body)
         vars conts statics hp rd)).
    + assert (Hcode :
        ∀ done : list static_binding,
          through_inv
            (State (Let_static (done ++ bindings) body) vars conts statics hp
               rd)
          ->
          refines
            (State (Let_static (done ++ bindings) body) vars conts statics hp
               rd)
            (State
               (Let_static
                  (done
                   ++ map (know_through_in_binding sets codes seen) bindings)
                  body)
               vars conts statics hp rd)).
      { clear Hinv.
        induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          intros done Hdone; cbn [map]; [ apply refines_refl | | | ].
        - cbn [know_through_in_binding know_through_in_code].
          destruct (cm_is_stub metadata || among my_closure params)
            eqn:Hleft.
          { (* The code is left. *)
            specialize
              (IH
                 (done
                  ++ [Bind_code id
                        (Code params my_closure ret exn metadata code_body)])).
            rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone). }
          apply orb_false_iff in Hleft as [_ Hmine].
          set (code_body' :=
                 know_through_in_body sets codes seen [(my_closure, id)] []
                   code_body).
          assert (Hdone' :
            through_inv
              (State
                 (Let_static
                    (done
                     ++ Bind_code id
                          (Code params my_closure ret exn metadata code_body')
                     :: bindings)
                    body)
                 vars conts statics hp rd)).
          { revert Hdone. apply through_inv_expr.
            - cbn [sets_made]. rewrite !flat_map_app.
              cbn [flat_map binding_sets_made code_sets_made].
              unfold code_body'. rewrite know_through_in_body_sets_made.
              reflexivity.
            - apply groups_whole_let_static_code. unfold code_body'.
              symmetry. apply know_through_in_body_groups_whole.
            - intros sight. cbn [sets_through]. rewrite !flat_map_app.
              cbn [flat_map binding_sets_through code_sets_through].
              unfold code_body'. rewrite know_through_in_body_sets_through.
              reflexivity.
            - intros sight. cbn [sets_seen]. rewrite !flat_map_app.
              cbn [flat_map binding_sets_seen code_sets_seen].
              unfold code_body'. rewrite know_through_in_body_sets_seen.
              reflexivity. }
          apply (refines_trans _
            (State
               (Let_static
                  (done
                   ++ Bind_code id
                        (Code params my_closure ret exn metadata code_body')
                   :: bindings)
                  body)
               vars conts statics hp rd)).
          + apply (let_static_code_refines_from_called through_inv done id
                     params my_closure ret exn metadata code_body code_body'
                     bindings body vars conts statics hp rd through_inv_kept
                     Hdone Hdone').
            * symmetry. unfold code_body'.
              apply know_through_in_body_binds_statics.
            * intros vs callee callee_vars return_to exn_to statics' hp' rd'
                value_slots Hbind _ _ Hclosure.
              split.
              -- apply through_inv_expr; unfold code_body';
                   [ apply know_through_in_body_sets_made
                   | apply know_through_in_body_groups_whole
                   | intros sight; apply know_through_in_body_sets_through
                   | intros sight; apply know_through_in_body_sets_seen ].
              -- intros Hentry.
                 apply know_through_in_body_refines;
                   [ exact Hentry | | apply paths_hold_nil ].
                 intros x c Hx. cbn [lookup] in Hx.
                 destruct (Pos.eqb_spec x my_closure) as [-> | _];
                   [ | discriminate ].
                 injection Hx as <-. exists callee, value_slots.
                 split; [ | exact Hclosure ].
                 rewrite (bind_params_lookup_other my_closure params vs _
                            callee_vars Hmine Hbind).
                 cbn [lookup]. rewrite Pos.eqb_refl. reflexivity.
          + specialize
              (IH
                 (done
                  ++ [Bind_code id
                        (Code params my_closure ret exn metadata code_body')])).
            rewrite <- !app_assoc in IH. cbn [app] in IH.
            exact (IH Hdone').
        - cbn [know_through_in_binding].
          specialize (IH (done ++ [Bind_deleted_code id])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone).
        - cbn [know_through_in_binding].
          specialize (IH (done ++ [Bind_symbol sym const])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone). }
      exact (Hcode [] Hinv).
    + assert (Hinv' :
        through_inv
          (State
             (Let_static
                (map (know_through_in_binding sets codes seen) bindings) body)
             vars conts statics hp rd)).
      { revert Hinv. apply through_inv_expr.
        - cbn [sets_made]. rewrite know_through_in_bindings_sets_made.
          reflexivity.
        - cbn [groups_whole].
          rewrite know_through_in_bindings_whole,
            know_through_in_bindings_groups_whole.
          reflexivity.
        - intros sight. cbn [sets_through].
          rewrite know_through_in_bindings_sets_through. reflexivity.
        - intros sight. cbn [sets_seen].
          rewrite know_through_in_bindings_sets_seen. reflexivity. }
      apply let_static_refines_step.
      destruct
        (eval_static_bindings vars
           (map (know_through_in_binding sets codes seen) bindings) statics)
        as [extended | ] eqn:Heval;
        [ | exact I ].
      apply know_code_through_refines_aux. apply (through_inv_kept _ _ Hinv').
      rewrite step_let_static, Heval. reflexivity.
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler].
    cbn [know_code_through_in_handler].
    assert (Hinv' :
      through_inv
        (State
           (Let_cont_non_recursive k
              (Handler params metadata
                 (know_code_through sets codes seen handler))
              body)
           vars conts statics hp rd)).
    { revert Hinv. apply through_inv_expr.
      - cbn [sets_made handler_sets_made].
        rewrite know_code_through_sets_made. reflexivity.
      - cbn [groups_whole handler_groups_whole].
        rewrite know_code_through_groups_whole. reflexivity.
      - intros sight. cbn [sets_through handler_sets_through].
        rewrite know_code_through_sets_through. reflexivity.
      - intros sight. cbn [sets_seen handler_sets_seen].
        rewrite know_code_through_sets_seen. reflexivity. }
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (know_code_through sets codes seen handler))
            body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_from through_inv k params metadata handler
               (know_code_through sets codes seen handler) body vars conts
               statics hp rd_ through_inv_kept Hinv Hinv').
      intros vs handler_vars conts' statics' hp' rd' _ _. split.
      * apply through_inv_expr;
          [ apply know_code_through_sets_made
          | apply know_code_through_groups_whole
          | intros sight; apply know_code_through_sets_through
          | intros sight; apply know_code_through_sets_seen ].
      * intros Hentry. apply know_code_through_refines_aux. exact Hentry.
    + apply let_cont_non_recursive_refines_step.
      apply know_code_through_refines_aux. apply (through_inv_kept _ _ Hinv').
      apply step_let_cont_non_recursive.
  - (* Let_cont_recursive *)
    set (know :=
           fun kh : continuation * continuation_handler =>
             let '(k, h) := kh in
             (k, know_code_through_in_handler sets codes seen h)).
    apply (refines_trans _
      (State (Let_cont_recursive (map know handlers) body) vars conts
         statics hp rd)).
    + assert (Hhandlers :
        ∀ done : list (continuation * continuation_handler),
          through_inv
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp rd)
          ->
          refines
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp rd)
            (State (Let_cont_recursive (done ++ map know handlers) body)
               vars conts statics hp rd)).
      { clear Hinv.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          intros done Hdone; cbn [map]; [ apply refines_refl | ].
        unfold know at 1. cbn [know_code_through_in_handler].
        assert (Hdone' :
          through_inv
            (State
               (Let_cont_recursive
                  (done
                   ++ (k,
                       Handler params metadata
                         (know_code_through sets codes seen handler))
                   :: handlers)
                  body)
               vars conts statics hp rd)).
        { revert Hdone. apply through_inv_expr.
          - cbn [sets_made]. rewrite !flat_map_app.
            cbn [flat_map handler_sets_made].
            rewrite know_code_through_sets_made. reflexivity.
          - cbn [groups_whole]. rewrite !forallb_app.
            cbn [forallb handler_groups_whole].
            rewrite know_code_through_groups_whole. reflexivity.
          - intros sight. cbn [sets_through]. rewrite !flat_map_app.
            cbn [flat_map handler_sets_through].
            rewrite know_code_through_sets_through. reflexivity.
          - intros sight. cbn [sets_seen]. rewrite !flat_map_app.
            cbn [flat_map handler_sets_seen].
            rewrite know_code_through_sets_seen. reflexivity. }
        apply (refines_trans _
          (State
             (Let_cont_recursive
                (done
                 ++ (k,
                     Handler params metadata
                       (know_code_through sets codes seen handler))
                 :: handlers)
                body)
             vars conts statics hp rd)).
        - apply (let_cont_recursive_handler_refines_from through_inv done k
                   params metadata handler
                   (know_code_through sets codes seen handler) handlers body
                   vars conts statics hp rd through_inv_kept Hdone Hdone').
          + symmetry. apply know_code_through_binds_statics.
          + intros vs handler_vars conts' statics' hp' rd' _ _. split.
            * apply through_inv_expr;
                [ apply know_code_through_sets_made
                | apply know_code_through_groups_whole
                | intros sight; apply know_code_through_sets_through
                | intros sight; apply know_code_through_sets_seen ].
            * intros Hentry. apply know_code_through_refines_aux.
              exact Hentry.
        - specialize
            (IH
               (done
                ++ [(k,
                     Handler params metadata
                       (know_code_through sets codes seen handler))])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone'). }
      exact (Hhandlers [] Hinv).
    + assert (Hinv' :
        through_inv
          (State (Let_cont_recursive (map know handlers) body) vars conts
             statics hp rd)).
      { revert Hinv. apply through_inv_expr.
        - cbn [sets_made]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
            [ reflexivity | ].
          unfold know at 1. rewrite know_code_through_in_handler_sets_made, IH.
          reflexivity.
        - cbn [groups_whole]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
            [ reflexivity | ].
          unfold know at 1.
          rewrite know_code_through_in_handler_groups_whole, IH.
          reflexivity.
        - intros sight. cbn [sets_through]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
            [ reflexivity | ].
          unfold know at 1.
          rewrite know_code_through_in_handler_sets_through, IH.
          reflexivity.
        - intros sight. cbn [sets_seen]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
            [ reflexivity | ].
          unfold know at 1.
          rewrite know_code_through_in_handler_sets_seen, IH.
          reflexivity. }
      apply let_cont_recursive_refines_step; [ reflexivity | ].
      intros Hbinds. apply know_code_through_refines_aux.
      apply (through_inv_kept _ _ Hinv').
      rewrite step_let_cont_recursive, Hbinds. reflexivity.
Qed.

(* The states that a unit starts in. *)
Lemma through_facts_hold_from (e : expr) (conts : cont_env)
    (statics : static_env) :
  incl (sets_through [] e) seen ->
  (∀ (c : code_id) (f : function_slot) (a b d : value_slot),
    conts_slots_through c f a b d conts
    ∧ conts_values_all (value_slots_through c f a b d) conts
    ∧ code_slots_through c f a b d statics
    ∧ Forall
        (fun binding => allocated_slots_through c f a b d (snd binding))
        (static_values statics)) ->
  through_facts_hold (State e [] conts statics empty_heap no_reads).
Proof.
  intros Hincl Hrest c f a b d Htest.
  destruct (Hrest c f a b d) as (Hconts & Hvalues & Hcode & Hstatics).
  apply state_slots_through_from; try assumption.
  unfold sets_through_give in Htest |- *. rewrite forallb_forall in Htest |- *.
  intros set Hin. exact (Htest set (Hincl set Hin)).
Qed.

End Through_known.

(* The static environment of the imports, and what may follow a unit. *)
Lemma imports_slots_through (imports : list (symbol * import_class))
    (c : code_id) (f : function_slot) (a b d : value_slot) :
  code_slots_through c f a b d (imports_statics imports)
  ∧ Forall
      (fun binding => allocated_slots_through c f a b d (snd binding))
      (static_values (imports_statics imports)).
Proof.
  split.
  - intros id code Hlookup. discriminate Hlookup.
  - unfold imports_statics. cbn [static_values]. apply Forall_forall.
    intros [s allocated] Hin.
    apply in_map_iff in Hin as ([s' c'] & Heq & _).
    injection Heq as _ <-. exact I.
Qed.

Lemma named_no_code_sets_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (sight : made_sight) (syms : list symbol)
    (defining : named) :
  named_statics_in_scope syms [] defining = true ->
  sets_through_give c f a b d (named_sets_through sight defining) = true.
Proof.
  destruct defining as [ | | code slots | at_slot decls slots | | | ]; simpl;
    try reflexivity.
  - discriminate.
  - intros H. apply andb_true_iff in H as [Hdecls _].
    destruct decls as [ | decl decls]; [ reflexivity | ].
    simpl in Hdecls. discriminate Hdecls.
Qed.

Fixpoint observer_sets_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (e : expr) {struct e} :
  ∀ sight : made_sight,
    binds_no_static e = true ->
    expr_statics_in_scope [] [] e = true ->
    sets_through_give c f a b d (sets_through sight e) = true.
Proof.
  destruct e
    as [ x xkind defining body | bindings body | k h body | handlers body | ap
       | ac | scrutinee arms | ];
    intros sight; cbn [sets_through]; try (intros _ _; reflexivity).
  - intros Hno H. simpl in Hno, H. apply andb_true_iff in H as [Hd Hbody].
    rewrite sets_through_give_app,
      (named_no_code_sets_through c f a b d sight [] defining Hd).
    exact (observer_sets_through c f a b d body _ Hno Hbody).
  - discriminate.
  - destruct h as [params metadata handler]. cbn [handler_sets_through].
    intros Hno H. simpl in Hno, H.
    apply andb_true_iff in Hno as [Hno_h Hno_b].
    apply andb_true_iff in H as [Hh Hb].
    rewrite ?(no_static_handler_scope _ _ body Hno_b),
      ?(no_static_recursive_scope _ _ body Hno_b)
      in Hh.
    simpl in Hh.
    rewrite sets_through_give_app,
      (observer_sets_through c f a b d handler _ Hno_h Hh),
      (observer_sets_through c f a b d body _ Hno_b Hb).
    reflexivity.
  - intros Hno H. simpl in Hno, H.
    apply andb_true_iff in Hno as [Hno_hs Hno_b].
    apply andb_true_iff in H as [Hhs Hb].
    rewrite sets_through_give_app,
      (observer_sets_through c f a b d body _ Hno_b Hb), andb_true_r.
    rewrite ?(no_static_handler_scope _ _ body Hno_b),
      ?(no_static_recursive_scope _ _ body Hno_b)
      in Hhs.
    simpl in Hhs.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      [ reflexivity | ].
    simpl in Hno_hs, Hhs.
    apply andb_true_iff in Hno_hs as [Hno_h Hno_rest].
    apply andb_true_iff in Hhs as [Hh Hrest].
    cbn [flat_map handler_sets_through].
    rewrite sets_through_give_app,
      (observer_sets_through c f a b d handler _ Hno_h Hh).
    exact (IH Hno_rest Hrest).
Qed.

Lemma observers_slots_through (c : code_id) (f : function_slot)
    (a b d : value_slot) (slots : list value_slot)
    (conts : cont_env) :
  observers slots conts -> conts_slots_through c f a b d conts.
Proof.
  unfold observers, conts_slots_through.
  induction conts as [ | [k cont] conts IH]; simpl;
    [ intros _; exact I | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  split; [ | exact (IH Hrest) ].
  revert Hc.
  destruct cont
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; exact I ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (Hno & Hscope & _).
  split; [ | exact I ].
  exists []. split; [ apply made_sight_holds_nil | ].
  split; [ | reflexivity ].
  exact (observer_sets_through c f a b d body _ Hno Hscope).
Qed.

(* The invariant of the state that a unit starts in, from its imports
   and with what may follow it. *)
Lemma through_inv_observed (sets : list set_made) (codes : list set_seen)
    (seen : list set_through) (imports : list (symbol * import_class))
    (slots : list value_slot) (conts : cont_env) (e : expr) :
  incl (sets_made e) sets ->
  incl (sets_seen [] e) codes ->
  incl (sets_through [] e) seen ->
  groups_whole e = true ->
  observers slots conts ->
  through_inv sets codes seen
    (State e [] conts (imports_statics imports) empty_heap no_reads).
Proof.
  intros Hincl Hcodes Hseen Hwhole Hobservers. split.
  - exact (slots_inv_observed sets codes imports slots conts e Hincl Hcodes
             Hwhole Hobservers).
  - apply through_facts_hold_from; [ exact Hseen | ].
    intros c f a b d.
    destruct (imports_slots_through imports c f a b d) as [Hcode Hstatics].
    exact (conj (observers_slots_through c f a b d slots conts Hobservers)
             (conj (observers_values_all _ slots conts Hobservers)
                (conj Hcode Hstatics))).
Qed.

(* ======================================================================== *)
(* The statements of the rewrite                                            *)
(* ======================================================================== *)

Theorem know_code_through_program_refines :
  know_code_through_program_refines_statement.
Proof.
  intros sets codes seen e Hincl Hcodes Hseen Hwhole. unfold program_refines.
  apply know_code_through_refines_aux.
  exact (through_inv_observed sets codes seen [] [] [] e Hincl Hcodes Hseen
           Hwhole (Forall_nil _)).
Qed.

Theorem read_slots_through_program_refines :
  read_slots_through_program_refines_statement.
Proof.
  intros e. unfold read_slots_through, program_refines.
  destruct (groups_whole e) eqn:Hwhole; [ | apply refines_refl ].
  apply know_code_through_refines_aux.
  exact (through_inv_observed (sets_made e) (sets_seen [] e)
           (sets_through [] e) [] [] [] e (incl_refl _) (incl_refl _)
           (incl_refl _) Hwhole (Forall_nil _)).
Qed.

Theorem read_slots_through_refines_imports :
  read_slots_through_refines_imports_statement.
Proof.
  intros imports e. unfold unit_refines, read_slots_through.
  destruct (groups_whole e) eqn:Hwhole; [ | apply refines_refl ].
  apply know_code_through_refines_aux.
  exact (through_inv_observed (sets_made e) (sets_seen [] e)
           (sets_through [] e) imports [] [] e (incl_refl _) (incl_refl _)
           (incl_refl _) Hwhole (Forall_nil _)).
Qed.

Theorem read_slots_through_refines_observed :
  read_slots_through_refines_observed_statement.
Proof.
  intros imports slots conts e Hobservers. unfold read_slots_through.
  destruct (groups_whole e) eqn:Hwhole; [ | apply refines_refl ].
  apply know_code_through_refines_aux.
  exact (through_inv_observed (sets_made e) (sets_seen [] e)
           (sets_through [] e) imports slots conts e (incl_refl _)
           (incl_refl _) (incl_refl _) Hwhole Hobservers).
Qed.

(* ======================================================================== *)
(* The names of the result                                                  *)
(* ======================================================================== *)

(* The variables that the projections remembered are projected from are
   among the names given: every entry, one that another hides
   included. *)
Definition bases_among (names : list variable) (through : projections)
    : Prop :=
  ∀ (y x : variable) (f : function_slot) (a : value_slot),
    (y, (x, f, a)) ∈ through -> x ∈ names.

Lemma bases_among_more (names names' : list variable)
    (through : projections) :
  incl names names' -> bases_among names through -> bases_among names' through.
Proof.
  intros Hincl Hbases y x f a Hin. exact (Hincl x (Hbases y x f a Hin)).
Qed.

Lemma bases_among_without (names : list variable) (x : variable)
    (through : projections) :
  bases_among names through -> bases_among names (through_without x through).
Proof.
  intros Hbases y x0 f a Hin.
  destruct (in_through_without x through y x0 f a Hin) as (Hin' & _ & _).
  exact (Hbases y x0 f a Hin').
Qed.

Lemma bases_among_without_all (names : list variable)
    (xs : list variable) :
  ∀ through : projections,
    bases_among names through ->
    bases_among names (through_without_all xs through).
Proof.
  induction xs as [ | x xs IH]; intros through Hbases; simpl;
    [ exact Hbases | ].
  apply IH, bases_among_without, Hbases.
Qed.

Lemma bases_among_under_let (names : list variable) (through : projections)
    (x : variable) (d : named) :
  bases_among names through ->
  bases_among (x :: names ++ free_variables_of_named d)
    (through_under_let through x d).
Proof.
  intros Hbases.
  assert (Hforgot :
    bases_among (x :: names ++ free_variables_of_named d)
      (through_without x through)).
  { apply (bases_among_more names); [ | apply bases_among_without, Hbases ].
    intros z Hz. right. apply in_or_app. left. exact Hz. }
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | f a closure | move_from move_to closure];
    try exact Hforgot.
  destruct closure as [from | sym | k]; try exact Hforgot.
  cbn [through_under_let].
  destruct (Pos.eqb from x); [ exact Hforgot | ].
  intros y x0 f0 a0 [Heq | Hin]; [ | exact (Hforgot y x0 f0 a0 Hin) ].
  injection Heq as _ <- _ _. right. apply in_or_app. right. left. reflexivity.
Qed.

(* What is put in the place of a projection names the variable that
   the projection remembered was projected from. *)
Lemma read_through_free (seen : list set_through) (known : closure_codes)
    (through : projections) (names : list variable) (d : named) :
  bases_among names through ->
  incl (free_variables_of_named (read_through seen known through d))
    (free_variables_of_named d ++ names).
Proof.
  intros Hbases.
  destruct (read_through_cases seen known through d)
    as [-> | (g & f & slot & a & b & y & x & c & -> & Hy & _ & _ & ->)];
    [ apply incl_appl, incl_refl | ].
  intros z [<- | []]. apply in_or_app. right.
  exact (Hbases y x f a (lookup_in _ _ _ Hy)).
Qed.

Lemma scoped_handler_under_body (names params : list variable)
    (metadata : handler_metadata) (handler handler' : expr) :
  scoped_under names handler handler' ->
  scoped_handler_under names (Handler params metadata handler)
    (Handler params metadata handler').
Proof.
  intros Hunder Hclosed. scope_all.
  destruct (Hunder Hclosed) as (? & ? & ?).
  split; [ assumption | split; names ].
Qed.

Fixpoint know_through_in_body_scoped (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  ∀ (known : closure_codes) (through : projections) (names : list variable),
    bases_among names through ->
    scoped_under names e (know_through_in_body sets codes seen known through e)
with know_through_in_handler_scoped (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  ∀ (known : closure_codes) (through : projections) (names : list variable),
    bases_among names through ->
    scoped_handler_under names h
      (know_through_in_handler sets codes seen known through h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros known through names Hbases; cbn [know_through_in_body];
      try apply scoped_under_refl.
    + apply scoped_under_let; [ apply read_through_free, Hbases | ].
      apply know_through_in_body_scoped, bases_among_under_let, Hbases.
    + apply scoped_under_let_static, know_through_in_body_scoped.
      exact (bases_among_more names _ through (incl_appl _ (incl_refl _))
               Hbases).
    + apply scoped_under_let_cont_handler;
        [ apply know_through_in_handler_scoped, Hbases
        | apply know_through_in_body_scoped, Hbases ].
    + apply scoped_under_let_cont_recursive_handlers;
        [ | | apply know_through_in_body_scoped, Hbases ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * apply
          (scoped_handlers_under_map_keyed names
             (fun _ h =>
                know_through_in_handler sets codes seen known through h)
             handlers).
        induction handlers as [ | [k h] handlers IH];
          [ apply Forall_nil | ].
        apply Forall_cons; [ | exact IH ].
        cbn [fst snd]. apply know_through_in_handler_scoped, Hbases.
  - destruct h as [params metadata handler].
    intros known through names Hbases. cbn [know_through_in_handler].
    apply scoped_handler_under_body, know_through_in_body_scoped,
      bases_among_without_all, Hbases.
Qed.

Fixpoint know_code_through_scoped (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (e : expr) {struct e} :
  scoped e (know_code_through sets codes seen e)
with know_code_through_in_handler_scoped (sets : list set_made)
    (codes : list set_seen)
    (seen : list set_through) (h : continuation_handler) {struct h} :
  scoped_handler h (know_code_through_in_handler sets codes seen h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [know_code_through]; try apply scoped_refl.
    + apply scoped_let;
        [ apply incl_refl | apply know_code_through_scoped ].
    + apply scoped_let_static; [ | apply know_code_through_scoped ].
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map]; [ apply scoped_bindings_nil | | | ].
      * cbn [know_through_in_binding know_through_in_code].
        destruct (cm_is_stub metadata || among my_closure params);
          [ apply scoped_bindings_cons_same, IH | ].
        apply scoped_bindings_cons_code; [ | exact IH ].
        apply scoped_code_intro, scoped_under_none,
          know_through_in_body_scoped.
        intros y x f a [].
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_same, IH.
    + apply scoped_let_cont;
        [ apply know_code_through_in_handler_scoped
        | apply know_code_through_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply know_code_through_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply know_code_through_in_handler_scoped | exact IH ].
  - destruct h as [params metadata handler].
    cbn [know_code_through_in_handler].
    apply scoped_handler_intro, know_code_through_scoped.
Qed.

Theorem read_slots_through_names : read_slots_through_names_statement.
Proof.
  intros e. unfold read_slots_through.
  destruct (groups_whole e); [ | apply scoped_refl ].
  apply know_code_through_scoped.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The laws below are proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v, but for those of the tests, which rest
   on a renaming being injective. *)

#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Instance set_made_renaming : Renaming_of set_made :=
  rename_set_made.

#[local] Instance closure_codes_renaming : Renaming_of closure_codes :=
  rename_codes.

(* What is seen of a closure, renamed: the variables among what its
   value slots were given. *)
Definition rename_made_seen (r : renaming) (made : made_seen) : made_seen :=
  map (fun slot => (fst slot, option_map (rename_simple r) (snd slot))) made.

#[local] Instance made_seen_renaming : Renaming_of made_seen :=
  rename_made_seen.

(* What is in sight, renamed. *)
Definition rename_made_sight (r : renaming) (sight : made_sight)
    : made_sight :=
  map
    (fun entry =>
       (rename_variable r (fst entry), rename_made_seen r (snd entry)))
    sight.

#[local] Instance made_sight_renaming : Renaming_of made_sight :=
  rename_made_sight.

(* A set as it is seen, renamed. *)
Definition rename_slot_through (r : renaming)
    (entry : value_slot * slot_through_seen)
    : value_slot * slot_through_seen :=
  (fst entry,
   (rename_simple r (fst (snd entry)),
    option_map (rename_made_seen r) (snd (snd entry)))).

Definition rename_set_through (r : renaming) (set : set_through)
    : set_through :=
  (fst set, map (rename_slot_through r) (snd set)).

#[local] Instance set_through_renaming : Renaming_of set_through :=
  rename_set_through.

(* What is remembered, renamed: the two variables of each entry. *)
Definition rename_projections (r : renaming) (through : projections)
    : projections :=
  map
    (fun entry =>
       (rename_variable r (fst entry),
        (rename_variable r (fst (fst (snd entry))), snd (fst (snd entry)),
         snd (snd entry))))
    through.

#[local] Instance projections_renaming : Renaming_of projections :=
  rename_projections.

Lemma var_in_simple_renamed_through (r : renaming) (x : variable)
    (s : simple) :
  var_in_simple (rename_variable r x) (rename_simple r s) = var_in_simple x s.
Proof.
  destruct s as [y | sym | k]; simpl;
    [ apply rename_variable_eqb | reflexivity | reflexivity ].
Qed.

Lemma forget_simple_renamed (r : renaming) (x : variable)
    (given : option simple) :
  forget_simple (rename_variable r x) (option_map (rename_simple r) given)
  = option_map (rename_simple r) (forget_simple x given).
Proof.
  destruct given as [s | ]; simpl; [ | reflexivity ].
  rewrite var_in_simple_renamed_through.
  destruct (var_in_simple x s); reflexivity.
Qed.

Lemma forget_in_made_renamed (r : renaming) (x : variable)
    (made : made_seen) :
  forget_in_made (rename_variable r x) (rename_made_seen r made)
  = rename_made_seen r (forget_in_made x made).
Proof.
  unfold forget_in_made, rename_made_seen, map_values. rewrite !map_map.
  apply map_ext. intros [slot given]. cbn [fst snd].
  rewrite forget_simple_renamed. reflexivity.
Qed.

Lemma unbind_made_sight_renamed (r : renaming) (x : variable)
    (sight : made_sight) :
  unbind (rename_variable r x) (rename_made_sight r sight)
  = rename_made_sight r (unbind x sight).
Proof.
  unfold unbind, rename_made_sight.
  induction sight as [ | [y made] sight IH]; [ reflexivity | ].
  cbn [map filter fst snd]. rewrite rename_variable_eqb, IH.
  destruct (Pos.eqb x y); reflexivity.
Qed.

Lemma made_without_renamed (r : renaming) (x : variable)
    (sight : made_sight) :
  made_without (rename_variable r x) (rename_made_sight r sight)
  = rename_made_sight r (made_without x sight).
Proof.
  unfold made_without. rewrite unbind_made_sight_renamed.
  unfold rename_made_sight, map_values. rewrite !map_map. apply map_ext.
  intros [y made]. cbn [fst snd]. rewrite forget_in_made_renamed.
  reflexivity.
Qed.

Lemma made_without_all_renamed (r : renaming) (xs : list variable) :
  ∀ sight : made_sight,
    made_without_all (map (rename_variable r) xs) (rename_made_sight r sight)
    = rename_made_sight r (made_without_all xs sight).
Proof.
  induction xs as [ | x xs IH]; intros sight; cbn [map made_without_all];
    [ reflexivity | ].
  rewrite made_without_renamed. apply IH.
Qed.

Lemma slots_made_renamed (r : renaming) (d : named) :
  slots_made (rename_named r d)
  = option_map (rename_made_seen r) (slots_made d).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | g slot closure | move_from move_to closure];
    try reflexivity;
    cbn [rename_named slots_made option_map]; f_equal;
    unfold rename_made_seen, map_values; rewrite !map_map; apply map_ext;
    intros [slot s]; reflexivity.
Qed.

Lemma made_under_let_renamed (r : renaming) (sight : made_sight)
    (x : variable) (d : named) :
  made_under_let (rename_made_sight r sight) (rename_variable r x)
    (rename_named r d)
  = rename_made_sight r (made_under_let sight x d).
Proof.
  unfold made_under_let. rewrite slots_made_renamed, made_without_renamed.
  destruct (slots_made d) as [made | ]; cbn [option_map]; [ | reflexivity ].
  rewrite forget_in_made_renamed. reflexivity.
Qed.

Lemma lookup_made_sight_renamed (r : renaming) (y : variable)
    (sight : made_sight) :
  lookup (rename_variable r y) (rename_made_sight r sight)
  = option_map (rename_made_seen r) (lookup y sight).
Proof.
  unfold rename_made_sight.
  induction sight as [ | [z made] sight IH]; [ reflexivity | ].
  cbn [map lookup fst snd]. rewrite rename_variable_eqb.
  destruct (Pos.eqb y z); [ reflexivity | exact IH ].
Qed.

Lemma simple_made_seen_renamed (r : renaming) (sight : made_sight)
    (s : simple) :
  simple_made_seen (rename_made_sight r sight) (rename_simple r s)
  = option_map (rename_made_seen r) (simple_made_seen sight s).
Proof.
  destruct s as [y | sym | k]; simpl;
    [ apply lookup_made_sight_renamed | reflexivity | reflexivity ].
Qed.

Lemma slots_through_renamed (r : renaming) (sight : made_sight)
    (value_slots : list (value_slot * simple)) :
  slots_through (rename_made_sight r sight)
    (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
  = map (rename_slot_through r) (slots_through sight value_slots).
Proof.
  unfold slots_through. rewrite !map_map. apply map_ext. intros [slot s].
  unfold rename_slot_through. cbn [fst snd].
  rewrite simple_made_seen_renamed. reflexivity.
Qed.

Lemma named_sets_through_renamed (r : renaming) (sight : made_sight)
    (d : named) :
  named_sets_through (rename_made_sight r sight) (rename_named r d)
  = map (rename_set_through r) (named_sets_through sight d).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | g slot closure | move_from move_to closure];
    try reflexivity;
    cbn [rename_named named_sets_through map]; unfold rename_set_through;
    cbn [fst snd]; rewrite slots_through_renamed; reflexivity.
Qed.

Lemma static_const_sets_through_renamed (r : renaming) (sight : made_sight)
    (const : static_const) :
  static_const_sets_through (rename_made_sight r sight)
    (rename_static_const r const)
  = map (rename_set_through r) (static_const_sets_through sight const).
Proof.
  destruct const
    as [code slots | slot closure_symbols decls slots | tag fields
       | contents | contents | elements];
    try reflexivity;
    cbn [rename_static_const static_const_sets_through map];
    unfold rename_set_through; cbn [fst snd]; rewrite slots_through_renamed;
    reflexivity.
Qed.

#[local] Hint Rewrite made_under_let_renamed made_without_all_renamed
  named_sets_through_renamed static_const_sets_through_renamed
  : equivariance.

Lemma sets_through_renamed (r : renaming) (e : expr) :
  ∀ sight : made_sight,
    sets_through (rename_made_sight r sight) (rename_expr r e)
    = map (rename_set_through r) (sets_through sight e).
Proof. equivariance. Qed.

Lemma sets_through_renamed_nil (r : renaming) (e : expr) :
  sets_through [] (rename_expr r e)
  = map (rename_set_through r) (sets_through [] e).
Proof. exact (sets_through_renamed r e []). Qed.

(* The tests answer the same of the sets renamed: a renaming is
   injective. *)
Lemma same_simple_renamed_through (r : renaming) (s s' : simple) :
  same_simple (rename_simple r s) (rename_simple r s') = same_simple s s'.
Proof.
  unfold same_simple.
  destruct (FlambdaModel.simple_eq_dec s s') as [Hsame | Hne],
    (FlambdaModel.simple_eq_dec (rename_simple r s) (rename_simple r s'))
      as [Heq | Hne'];
    try reflexivity.
  - exfalso. apply Hne'. rewrite Hsame. reflexivity.
  - exfalso. apply Hne. exact (whole_rename_simple_injective r s s' Heq).
Qed.

Lemma lookup_rename_slot_through (r : renaming) (slot : value_slot)
    (slots : list (value_slot * slot_through_seen)) :
  lookup slot (map (rename_slot_through r) slots)
  = option_map
      (fun seen : slot_through_seen =>
         (rename_simple r (fst seen),
          option_map (rename_made_seen r) (snd seen)))
      (lookup slot slots).
Proof.
  induction slots as [ | [slot' seen'] slots IH]; [ reflexivity | ].
  cbn [map lookup rename_slot_through fst snd].
  destruct (Pos.eqb slot slot'); [ reflexivity | exact IH ].
Qed.

Lemma lookup_rename_made_seen (r : renaming) (slot : value_slot)
    (made : made_seen) :
  lookup slot (rename_made_seen r made)
  = option_map (option_map (rename_simple r)) (lookup slot made).
Proof.
  unfold rename_made_seen.
  induction made as [ | [slot' given] made IH]; [ reflexivity | ].
  cbn [map lookup fst snd].
  destruct (Pos.eqb slot slot'); [ reflexivity | exact IH ].
Qed.

Lemma set_through_gives_renamed (r : renaming) (a b d : value_slot)
    (set : set_through) :
  set_through_gives a b d (rename_set_through r set)
  = set_through_gives a b d set.
Proof.
  unfold set_through_gives, rename_set_through. cbn [snd].
  rewrite !lookup_rename_slot_through.
  destruct (lookup a (snd set)) as [[sa [made | ]] | ];
    cbn [option_map fst snd]; try reflexivity.
  destruct (lookup b (snd set)) as [[held seen_b] | ];
    cbn [option_map fst snd]; try reflexivity.
  rewrite lookup_rename_made_seen.
  destruct (lookup d made) as [[inner | ] | ]; cbn [option_map];
    try reflexivity.
  apply same_simple_renamed_through.
Qed.

Lemma sets_through_give_renamed (r : renaming) (c : code_id)
    (f : function_slot) (a b d : value_slot) (seen : list set_through) :
  sets_through_give c f a b d (map (rename_set_through r) seen)
  = sets_through_give c f a b d seen.
Proof.
  unfold sets_through_give.
  induction seen as [ | set seen IH]; cbn [map forallb]; [ reflexivity | ].
  rewrite set_through_gives_renamed, IH. reflexivity.
Qed.

Lemma slot_through_renamed (r : renaming) (seen : list set_through)
    (c : code_id) (f : function_slot) (a d : value_slot) :
  slot_through (map (rename_set_through r) seen) c f a d
  = slot_through seen c f a d.
Proof.
  unfold slot_through.
  assert (Hfind :
    find (fun set => has_code_at c f (fst set))
      (map (rename_set_through r) seen)
    = option_map (rename_set_through r)
        (find (fun set => has_code_at c f (fst set)) seen)).
  { induction seen as [ | set seen IH]; [ reflexivity | ].
    cbn [map find rename_set_through fst].
    destruct (has_code_at c f (fst set)); [ reflexivity | exact IH ]. }
  rewrite Hfind.
  destruct (find (fun set => has_code_at c f (fst set)) seen) as [set | ];
    cbn [option_map]; [ | reflexivity ].
  cbn [rename_set_through snd].
  generalize (snd set). intros slots.
  induction slots as [ | [slot seen_slot] slots IH]; [ reflexivity | ].
  cbn [map find rename_slot_through fst snd].
  rewrite sets_through_give_renamed.
  destruct (sets_through_give c f a slot d seen); [ reflexivity | exact IH ].
Qed.

Lemma lookup_projections_renamed (r : renaming) (y : variable)
    (through : projections) :
  lookup (rename_variable r y) (rename_projections r through)
  = option_map
      (fun entry : variable * function_slot * value_slot =>
         (rename_variable r (fst (fst entry)), snd (fst entry), snd entry))
      (lookup y through).
Proof.
  unfold rename_projections.
  induction through as [ | [z [[x f] a]] through IH]; [ reflexivity | ].
  cbn [map lookup fst snd]. rewrite rename_variable_eqb.
  destruct (Pos.eqb y z); [ reflexivity | exact IH ].
Qed.

Lemma read_through_renamed (r : renaming) (seen : list set_through)
    (known : closure_codes) (through : projections) (d : named) :
  read_through (map (rename_set_through r) seen) (rename_codes r known)
    (rename_projections r through) (rename_named r d)
  = rename_named r (read_through seen known through d).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | g slot closure | move_from move_to closure];
    try reflexivity.
  destruct closure as [y | sym | k]; try reflexivity.
  cbn [rename_named rename_simple read_through].
  rewrite lookup_projections_renamed.
  destruct (lookup y through) as [[[x f] a] | ]; cbn [option_map fst snd];
    [ | reflexivity ].
  rewrite lookup_codes_renamed.
  destruct (lookup x known) as [c | ]; [ | reflexivity ].
  rewrite slot_through_renamed.
  destruct (slot_through seen c f a slot) as [b | ]; reflexivity.
Qed.

Lemma through_without_renamed (r : renaming) (x : variable)
    (through : projections) :
  through_without (rename_variable r x) (rename_projections r through)
  = rename_projections r (through_without x through).
Proof.
  unfold through_without, unbind, rename_projections.
  induction through as [ | [y [[x0 f] a]] through IH]; [ reflexivity | ].
  cbn [map filter fst snd]. rewrite rename_variable_eqb.
  destruct (Pos.eqb x y); cbn [negb]; [ exact IH | ].
  cbn [filter fst snd]. rewrite rename_variable_eqb.
  destruct (Pos.eqb x x0); cbn [negb map fst snd];
    [ exact IH | rewrite IH; reflexivity ].
Qed.

Lemma through_without_all_renamed (r : renaming) (xs : list variable) :
  ∀ through : projections,
    through_without_all (map (rename_variable r) xs)
      (rename_projections r through)
    = rename_projections r (through_without_all xs through).
Proof.
  induction xs as [ | x xs IH]; intros through;
    cbn [map through_without_all]; [ reflexivity | ].
  rewrite through_without_renamed. apply IH.
Qed.

Lemma through_under_let_renamed (r : renaming) (through : projections)
    (x : variable) (d : named) :
  through_under_let (rename_projections r through) (rename_variable r x)
    (rename_named r d)
  = rename_projections r (through_under_let through x d).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | f a closure | move_from move_to closure];
    cbn [rename_named through_under_let];
    try apply through_without_renamed.
  destruct closure as [from | sym | k];
    cbn [rename_simple through_under_let];
    try apply through_without_renamed.
  rewrite rename_variable_eqb, through_without_renamed.
  destruct (Pos.eqb from x); reflexivity.
Qed.

#[local] Hint Rewrite set_made_has_renamed sets_made_have_renamed
  using (solve [ assumption | reflexivity ]) : equivariance.

#[local] Hint Rewrite lookup_codes_renamed unbind_codes_renamed
  set_with_code_renamed read_through_renamed through_under_let_renamed
  through_without_all_renamed : equivariance.

(* The sets as FlambdaSlotsOfCodeSpec.sets_seen sees them name no
   variable: they are the same of a term renamed. *)
Lemma know_code_through_renamed (r : renaming) (sets : list set_made)
    (codes : list set_seen) (seen : list set_through) (e : expr) :
  know_code_through (map (rename_set_made r) sets) codes
    (map (rename_set_through r) seen) (rename_expr r e)
  = rename_expr r (know_code_through sets codes seen e).
Proof. equivariance. Qed.

#[local] Hint Rewrite groups_whole_renamed sets_made_renamed
  sets_seen_renamed_nil sets_through_renamed_nil know_code_through_renamed
  : equivariance.

Theorem read_slots_through_equivariant :
  read_slots_through_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* With what else is known of the closures of the code of a unit            *)
(* ======================================================================== *)

Theorem know_own_closures_through_program_refines :
  know_own_closures_through_program_refines_statement.
Proof.
  intros e. unfold know_own_closures_through, program_refines.
  exact (refines_trans _ _ _ (know_own_closures_program_refines e)
           (read_slots_through_program_refines (know_own_closures e))).
Qed.

Theorem know_own_closures_through_refines_imports :
  know_own_closures_through_refines_imports_statement.
Proof.
  intros imports e. unfold know_own_closures_through, unit_refines.
  exact (refines_trans _ _ _ (know_own_closures_refines_imports imports e)
           (read_slots_through_refines_imports imports (know_own_closures e))).
Qed.

Theorem know_own_closures_through_refines_observed :
  know_own_closures_through_refines_observed_statement.
Proof.
  intros imports slots conts e Hobservers. unfold know_own_closures_through.
  exact (refines_trans _ _ _
           (know_own_closures_refines_observed imports slots conts e
              Hobservers)
           (read_slots_through_refines_observed imports slots conts
              (know_own_closures e) Hobservers)).
Qed.

Theorem know_own_closures_through_names :
  know_own_closures_through_names_statement.
Proof.
  intros e. unfold know_own_closures_through.
  exact (scoped_trans e (know_own_closures e) _ (know_own_closures_names e)
           (read_slots_through_names (know_own_closures e))).
Qed.

Theorem know_own_closures_through_equivariant :
  know_own_closures_through_equivariant_statement.
Proof.
  intros r e. unfold know_own_closures_through.
  rewrite know_own_closures_equivariant. apply read_slots_through_equivariant.
Qed.

(* ======================================================================== *)
(* The examples                                                             *)
(* ======================================================================== *)

Theorem read_slots_through_reads : read_slots_through_reads_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_constant : read_slots_through_constant_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_symbol : read_slots_through_symbol_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_two_places :
  read_slots_through_two_places_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_handler : read_slots_through_handler_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_loop : read_slots_through_loop_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_in_code : read_slots_through_in_code_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_other_function_slot :
  read_slots_through_other_function_slot_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_other : read_slots_through_other_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_alias : read_slots_through_alias_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_one_place : read_slots_through_one_place_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_bound_between :
  read_slots_through_bound_between_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_own_binder :
  read_slots_through_own_binder_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_parameter : read_slots_through_parameter_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_carried : read_slots_through_carried_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_twice : read_slots_through_twice_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_twice_forgotten :
  read_slots_through_twice_forgotten_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_no_slot : read_slots_through_no_slot_statement.
Proof. vm_compute. split; reflexivity. Qed.

Theorem read_slots_through_projected : read_slots_through_projected_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_lifted : read_slots_through_lifted_statement.
Proof. vm_compute. split; reflexivity. Qed.

Theorem read_slots_through_reads_own_binder :
  read_slots_through_reads_own_binder_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_base_bound :
  read_slots_through_base_bound_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_held : read_slots_through_held_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_held_alias :
  read_slots_through_held_alias_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_held_two_codes :
  read_slots_through_held_two_codes_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem read_slots_through_held_symbol :
  read_slots_through_held_symbol_statement.
Proof. vm_compute. repeat split; reflexivity. Qed.

Theorem read_slots_through_run : read_slots_through_run_statement.
Proof.
  intros e [<- | [<- | []]]; exists 20; eexists; vm_compute; reflexivity.
Qed.

Theorem read_slots_through_other_function_slot_run :
  read_slots_through_other_function_slot_run_statement.
Proof.
  split.
  - exists 20. vm_compute. reflexivity.
  - exists 20. eexists. vm_compute. reflexivity.
Qed.

Theorem read_slots_through_held_run : read_slots_through_held_run_statement.
Proof.
  intros e [<- | [<- | []]]; exists 30; eexists; vm_compute; reflexivity.
Qed.
