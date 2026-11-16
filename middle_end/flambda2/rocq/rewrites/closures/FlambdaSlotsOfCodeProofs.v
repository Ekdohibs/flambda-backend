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

(* Proofs of the statements of FlambdaSlotsOfCodeSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Arith Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaKnownSymbolsSpec FlambdaClosuresSpec
  FlambdaClosuresProofs FlambdaContextWhereSpec FlambdaKnownSlotsSpec
  FlambdaKnownSlotsProofs FlambdaSetsOfCodeSpec FlambdaSetsOfCodeProofs
  FlambdaSlotsOfCodeSpec.
Import ListNotations.

(* ======================================================================== *)
(* Where a call, or the application of a continuation, lands                *)
(* ======================================================================== *)

(* As the section of that name of FlambdaSemanticsProofs.v, with one
   thing more said of the handler that the application of a
   continuation lands on: the variables that it runs under are those
   that its closure keeps with its parameters bound. *)
Section Lands_bound.

Variable statics : static_env.
Variable hp : heap.
Variable rd : read_record.
Variable Kc : cont_value -> Prop.
Variable Q : state -> Prop.

Hypothesis Kc_extra :
  ∀ (c : cont_value) (extra : list value), Kc (Extra_args c extra) -> Kc c.

Hypothesis Kc_over :
  ∀ (remaining : list value) (remaining_arity return_arity : arity)
    (return_to exn_to : cont_value),
    Kc
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
    <-> Kc return_to ∧ Kc exn_to.

Hypothesis Q_code :
  ∀ (called : code_id) (c : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some c ->
    Kc return_to ->
    Kc exn_to ->
    Q (State (code_body c) callee_vars
         [(code_return_continuation c, return_to);
          (code_exn_continuation c, exn_to)]
         statics hp rd).

Hypothesis Q_handler :
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (handler_vars : env),
    Kc
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    bind_params params vs def_vars = Some handler_vars ->
    Q (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).

Lemma call_code_lands_bound (closure : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (s' : state) :
  lookup called (static_code statics) = Some c ->
  Kc return_to ->
  Kc exn_to ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s' ->
  Q s'.
Proof.
  intros Hcode Hreturn Hexn Hstep.
  apply call_code_step in Hstep as (callee_vars & _ & _ & _ & _ & _ & ->).
  exact (Q_code called c callee_vars return_to exn_to Hcode Hreturn Hexn).
Qed.

Lemma run_call_lands_bound (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (s' : state) :
  shape_calls statics kind shape ->
  Kc return_to ->
  Kc exn_to ->
  (∀ partial : value, return_partial partial = Step s' -> Q s') ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  Q s'.
Proof.
  intros Hcalls Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    exact (call_code_lands_bound closure called c vs _ _ _ _ s' Hcode Hreturn
             Hexn).
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    apply (call_code_lands_bound closure called c vs _ _ _ _ s' Hcode);
      [ apply Kc_over; split; assumption | exact Hexn ].
  - apply Hpartial.
  - discriminate.
Qed.

Lemma apply_cont_value_lands_bound (c : cont_value) :
  ∀ (vs : list value) (s' : state),
    Kc c -> apply_cont_value c vs statics hp rd = Step s' -> Q s'.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs s' Hc Hstep.
  - simpl in Hstep.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | discriminate ].
    injection Hstep as <-.
    exact (Q_handler params metadata handler def_vars def_conts recursive
             vs handler_vars Hc Hbind).
  - discriminate.
  - exact (IH _ s' (Kc_extra c extra Hc) Hstep).
  - rewrite apply_cont_value_over_application in Hstep.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    apply Kc_over in Hc as [Hreturn Hexn].
    unfold call_function in Hstep.
    apply (run_call_lands_bound _ Indirect_unknown_arity _ _ _ _ s'
             (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn)
      in Hstep; [ exact Hstep | ].
    intros partial. exact (IHreturn [partial] s' Hreturn).
Qed.

Lemma call_function_lands_bound (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (s' : state) :
  Kc return_to ->
  Kc exn_to ->
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Step s' ->
  Q s'.
Proof.
  intros Hreturn Hexn. unfold call_function.
  apply (run_call_lands_bound _ kind _ _ _ _ s'
           (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn).
  intros partial. exact (apply_cont_value_lands_bound return_to _ s' Hreturn).
Qed.

End Lands_bound.

(* ======================================================================== *)
(* The test, of the sets of the parts of a term                             *)
(* ======================================================================== *)

Lemma sets_seen_give_app (c : code_id) (f : function_slot) (slot : value_slot)
    (d : code_id) (seen seen' : list set_seen) :
  sets_seen_give c f slot d (seen ++ seen')
  = sets_seen_give c f slot d seen && sets_seen_give c f slot d seen'.
Proof. unfold sets_seen_give. apply forallb_app. Qed.

Lemma bindings_sets_seen_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (known : closure_sight)
    (bindings : list static_binding) :
  sets_seen_give c f slot d (flat_map (binding_sets_seen known) bindings)
  = forallb
      (fun binding =>
         sets_seen_give c f slot d (binding_sets_seen known binding))
      bindings.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite sets_seen_give_app, IH. reflexivity.
Qed.

Lemma handlers_sets_seen_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (known : closure_sight)
    (handlers : list (continuation * continuation_handler)) :
  sets_seen_give c f slot d
    (flat_map (fun '(_, h) => handler_sets_seen known h) handlers)
  = forallb
      (fun '(_, h) => sets_seen_give c f slot d (handler_sets_seen known h))
      handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite sets_seen_give_app, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* What is in sight holds                                                   *)
(* ======================================================================== *)

Lemma sight_holds_nil (vars : env) : sight_holds [] vars.
Proof. intros x seen H. discriminate H. Qed.

Lemma sight_holds_unbind (known : closure_sight) (vars : env) (x : variable)
    (v : value) :
  sight_holds known vars -> sight_holds (unbind x known) ((x, v) :: vars).
Proof.
  intros Hhold y seen Hy. rewrite lookup_unbind in Hy. cbn [lookup].
  destruct (Pos.eqb y x); [ discriminate | ].
  exact (Hhold y seen Hy).
Qed.

Lemma sight_holds_bind_params (params : list variable) :
  ∀ (known : closure_sight) (vs : list value) (vars handler_vars : env),
    sight_holds known vars ->
    bind_params params vs vars = Some handler_vars ->
    sight_holds (sight_without params known) handler_vars.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, sight_holds_unbind, Hhold.
Qed.

(* The closure that a defining expression is seen to give is the one
   that it gives. *)
Lemma sight_made_holds (statics : static_env) (known : closure_sight)
    (vars : env) (defining : named) (seen : closure_seen) (v : value) :
  sight_holds known vars ->
  sight_made known defining = Some seen ->
  eval_named statics vars defining = Some v ->
  closure_value_seen v = Some seen.
Proof.
  intros Hhold.
  destruct defining as [s | p args | code slots | at_slot decls slots
    | slot closure | project_from slot closure | move_from move_to closure];
    simpl; try discriminate.
  - intros [= <-].
    destruct (eval_value_slots vars slots) as [vs | ]; [ | discriminate ].
    simpl. intros [= <-]. reflexivity.
  - intros [= <-].
    destruct (lookup at_slot decls) as [code | ] eqn:Hslot;
      [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Hslot. reflexivity.
  - destruct closure as [x | sym | k]; try discriminate.
    destruct (lookup x known) as [[from_slot decls] | ] eqn:Hx;
      [ | discriminate ].
    intros [= <-].
    destruct (Hhold x _ Hx) as (w & Hw & Hseen). simpl. rewrite Hw. simpl.
    unfold project_function_slot.
    destruct w as [k | code slots | at_slot set_decls slots | s | tag fields
      | elements | contents | l | pa_callee pa_args pa_arity];
      simpl in Hseen; try discriminate.
    + injection Hseen as <- <-. intros Hproject.
      apply project_unnamed_function_slot in Hproject as (_ & -> & ->).
      reflexivity.
    + destruct (lookup at_slot set_decls) as [code | ]; [ | discriminate ].
      simpl in Hseen. injection Hseen as <- <-.
      destruct (Pos.eqb at_slot move_from); [ | discriminate ].
      destruct (lookup move_to set_decls) as [code' | ] eqn:Hto;
        [ | discriminate ].
      simpl. intros [= <-]. simpl. rewrite Hto. reflexivity.
Qed.

Lemma exec_named_sight (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (known : closure_sight) (vars : env)
    (defining : named) (seen : closure_seen) (v : value) :
  sight_holds known vars ->
  sight_made known defining = Some seen ->
  exec_named statics hp rd vars defining = Some (v, hp', rd') ->
  closure_value_seen v = Some seen.
Proof.
  intros Hhold Hmade Hexec.
  assert (Hnot : ∀ (p : prim) (args : list simple), defining <> Prim p args).
  { intros p args ->. discriminate Hmade. }
  rewrite (exec_named_not_prim _ _ rd_ _ _ Hnot) in Hexec.
  destruct (eval_named statics vars defining) as [w | ] eqn:Heval;
    [ | discriminate ].
  simpl in Hexec. injection Hexec as <- _ _.
  exact (sight_made_holds statics known vars defining seen w Hhold Hmade
           Heval).
Qed.

Lemma sight_holds_under_let (known : closure_sight) (vars : env)
    (x : variable) (defining : named) (v : value) :
  sight_holds known vars ->
  (∀ seen : closure_seen,
    sight_made known defining = Some seen ->
    closure_value_seen v = Some seen) ->
  sight_holds (sight_under_let known x defining) ((x, v) :: vars).
Proof.
  intros Hhold Hmade. unfold sight_under_let.
  destruct (sight_made known defining) as [seen | ].
  - intros y seen' Hy. cbn [lookup] in Hy |- *.
    destruct (Pos.eqb y x) eqn:Hyx.
    + injection Hy as <-. exists v. split; [ reflexivity | ].
      exact (Hmade seen eq_refl).
    + rewrite lookup_unbind, Hyx in Hy. exact (Hhold y seen' Hy).
  - exact (sight_holds_unbind known vars x v Hhold).
Qed.

(* ======================================================================== *)
(* A set that a term makes, and the set of the closure that it gives        *)
(* ======================================================================== *)

Lemma lookup_slots_seen (known : closure_sight) (slot : value_slot) :
  ∀ value_slots : list (value_slot * simple),
    lookup slot (slots_seen known value_slots)
    = match lookup slot value_slots with
      | Some a => Some (simple_seen_code known a)
      | None => None
      end.
Proof.
  induction value_slots as [ | [slot' a] value_slots IH]; simpl;
    [ reflexivity | ].
  destruct (Pos.eqb slot slot'); [ reflexivity | exact IH ].
Qed.

(* A set that a term makes under a sight that holds, and that passes
   the test, gives a closure whose set is so. *)
Lemma set_seen_held (c : code_id) (f : function_slot) (slot : value_slot)
    (d : code_id) (known : closure_sight) (vars : env)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) :
  sight_holds known vars ->
  eval_value_slots vars value_slots = Some slot_values ->
  sets_seen_give c f slot d [(function_decls, slots_seen known value_slots)]
  = true ->
  set_held_gives c f slot d function_decls slot_values = true.
Proof.
  intros Hhold Heval H. unfold sets_seen_give in H. cbn [forallb fst] in H.
  rewrite andb_true_r in H. unfold set_held_gives.
  apply orb_true_iff in H as [H | H]; [ rewrite H; reflexivity | ].
  apply orb_true_iff. right.
  unfold set_seen_gives in H. cbn [snd] in H.
  rewrite lookup_slots_seen in H.
  destruct (lookup slot value_slots) as [a | ] eqn:Ha; [ | discriminate ].
  destruct a as [y | sym | k]; simpl in H; try discriminate.
  destruct (lookup y known) as [seen | ] eqn:Hy; [ | discriminate ].
  destruct (seen_code seen) as [code | ] eqn:Hcode; [ | discriminate ].
  destruct (Hhold y seen Hy) as (w & Hw & Hseen).
  destruct (eval_value_slots_lookup vars value_slots slot_values slot (Var y)
              Heval Ha) as (v & -> & Hv).
  simpl in Hv. rewrite Hw in Hv. injection Hv as <-.
  unfold value_is_closure_of. rewrite Hseen, Hcode. exact H.
Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Section Slots_give.

Variable c : code_id.
Variable f : function_slot.
Variable slot : value_slot.
Variable d : code_id.

Let Pv (v : value) : Prop := value_slots_give c f slot d v = true.
Let Qv (allocated : statically_allocated_value) : Prop :=
  allocated_slots_give c f slot d allocated = true.

(* The values. *)
Lemma slot_values_slots_give_in (slot_values : list (value_slot * value)) :
  forallb (fun '(_, contents) => value_slots_give c f slot d contents)
    slot_values
  = true ->
  ∀ (w : value_slot) (contents : value),
    (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros H w contents Hin. rewrite forallb_forall in H.
  exact (H (w, contents) Hin).
Qed.

Lemma statics_slots_give_lookup (statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  lookup s (static_values statics) = Some allocated ->
  Qv allocated.
Proof.
  intros Hstatics Hs. rewrite Forall_forall in Hstatics.
  exact (Hstatics (s, allocated) (lookup_in _ _ _ Hs)).
Qed.

Lemma eval_simple_slots_give (vars : env) (s : simple) (v : value) :
  env_values_all Pv vars -> eval_simple vars s = Some v -> Pv v.
Proof.
  intros Hvars. destruct s as [x | sym | k]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

Lemma eval_simples_slots_give (vars : env) :
  env_values_all Pv vars ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs ->
    forallb (value_slots_give c f slot d) vs = true.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_slots_give vars s v Hvars Hs). exact (IH rest eq_refl).
Qed.

Lemma eval_value_slots_slots_give (vars : env) :
  env_values_all Pv vars ->
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)),
    eval_value_slots vars value_slots = Some slot_values ->
    forallb (fun '(_, contents) => value_slots_give c f slot d contents)
      slot_values
    = true.
Proof.
  intros Hvars.
  induction value_slots as [ | [slot' s] value_slots IH]; intros slot_values;
    simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_slots_give vars s v Hvars Hs). exact (IH rest eq_refl).
Qed.

Lemma values_slots_give_Forall (vs : list value) :
  forallb (value_slots_give c f slot d) vs = true -> Forall Pv vs.
Proof.
  intros H. rewrite forallb_forall in H. apply Forall_forall. exact H.
Qed.

(* A closure that is so, read as a closure of a set: its set is so, and
   the values of its value slots are so. *)
Lemma closure_in_set_slots_give (statics : static_env) (v : value)
    (at_slot : function_slot) (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  Pv v ->
  closure_in_set statics v = Some (at_slot, decls, slot_values) ->
  set_held_gives c f slot d decls slot_values = true
  ∧ ∀ (w : value_slot) (contents : value),
      (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros Hstatics Hv. unfold closure_in_set.
  destruct v as [k | code slots | at_slot' set_decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - unfold Pv in Hv. simpl in Hv. apply andb_true_iff in Hv as [Hset Hslots].
    intros [= <- <- <-].
    split; [ exact Hset | exact (slot_values_slots_give_in slots Hslots) ].
  - unfold Pv in Hv. simpl in Hv. apply andb_true_iff in Hv as [Hset Hslots].
    destruct (lookup at_slot' set_decls); [ | discriminate ].
    simpl. intros [= <- <- <-].
    split; [ exact Hset | exact (slot_values_slots_give_in slots Hslots) ].
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot' syms set_decls slots | tag fields | contents
          | bytes | elements | ] | ] eqn:Hs;
      try discriminate.
    + pose proof (statics_slots_give_lookup statics s _ Hstatics Hs) as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Hset Hslots].
      intros [= <- <- <-].
      split; [ exact Hset | exact (slot_values_slots_give_in slots Hslots) ].
    + pose proof (statics_slots_give_lookup statics s _ Hstatics Hs) as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Hset Hslots].
      destruct (lookup at_slot' set_decls); [ | discriminate ].
      simpl. intros [= <- <- <-].
      split; [ exact Hset | exact (slot_values_slots_give_in slots Hslots) ].
Qed.

Lemma value_slots_give_of_numbers : holds_of_numbers Pv.
Proof. intros k. split; reflexivity. Qed.

Lemma value_slots_give_of_blocks : holds_of_blocks Pv.
Proof.
  intros tag fields.
  exact (forallb_forall (value_slots_give c f slot d) fields).
Qed.

Lemma value_slots_give_of_arrays : holds_of_arrays Pv.
Proof.
  intros elements.
  exact (forallb_forall (value_slots_give c f slot d) elements).
Qed.

Lemma value_slots_give_in_static_blocks (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_blocks statics Pv.
Proof.
  intros Hstatics s tag fields _ Hs v Hin.
  pose proof (statics_slots_give_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid. rewrite forallb_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma value_slots_give_in_static_arrays (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_arrays statics Pv.
Proof.
  intros Hstatics s elements _ Hs v Hin.
  pose proof (statics_slots_give_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid. rewrite forallb_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma project_function_slot_slots_give (statics : static_env)
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

Lemma eval_named_slots_give (statics : static_env) (known : closure_sight)
    (vars : env) (defining : named) (v : value) :
  sight_holds known vars ->
  sets_seen_give c f slot d (named_sets_seen known defining) = true ->
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
  - exact (eval_simple_slots_give vars s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    simpl. intros Hprim.
    apply (eval_prim_preserves Pv statics p vs v value_slots_give_of_numbers
             value_slots_give_of_blocks value_slots_give_of_arrays
             (value_slots_give_in_static_blocks statics Hstatics)
             (value_slots_give_in_static_arrays statics Hstatics));
      [ | exact Hprim ].
    pose proof (eval_simples_slots_give vars Hvars args vs Hargs) as Hvs.
    rewrite forallb_forall in Hvs. exact Hvs.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl.
    cbn [named_sets_seen] in Hd.
    rewrite (set_seen_held c f slot d known vars _ slots vs Hhold Hslots Hd).
    exact (eval_value_slots_slots_give vars Hvars slots vs Hslots).
  - destruct (lookup at_slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl.
    cbn [named_sets_seen] in Hd.
    rewrite (set_seen_held c f slot d known vars _ slots vs Hhold Hslots Hd).
    exact (eval_value_slots_slots_give vars Hvars slots vs Hslots).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    destruct (closure_of statics k) as [[code slot_values] | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics k code slot_values) Hclosure)
      as (at_slot & decls & Hin_set & _).
    destruct (closure_in_set_slots_give statics k at_slot decls slot_values
                Hstatics (eval_simple_slots_give vars closure k Hvars Hk)
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
    destruct (closure_in_set_slots_give statics k at_slot decls slot_values
                Hstatics (eval_simple_slots_give vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot' v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    exact (project_function_slot_slots_give statics move_from move_to k v
             (eval_simple_slots_give vars closure k Hvars Hk)).
Qed.

Lemma eval_static_const_slots_give (known : closure_sight)
    (const : static_const) (vars : env)
    (allocated : statically_allocated_value) :
  sight_holds known vars ->
  sets_seen_give c f slot d (static_const_sets_seen known const) = true ->
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
    cbn [static_const_sets_seen] in Hconst.
    rewrite (set_seen_held c f slot d known vars _ slots vs Hhold Hslots
               Hconst).
    exact (eval_value_slots_slots_give vars Hvars slots vs Hslots).
  - destruct (lookup at_slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl.
    cbn [static_const_sets_seen] in Hconst.
    rewrite (set_seen_held c f slot d known vars _ slots vs Hhold Hslots
               Hconst).
    exact (eval_value_slots_slots_give vars Hvars slots vs Hslots).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_slots_give vars Hvars fields vs Hfields).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simple_slots_give vars contents v Hvars Hcontents).
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_slots_give vars Hvars elements vs Helements).
Qed.

Lemma eval_static_bindings_slots_give (known : closure_sight) (vars : env) :
  sight_holds known vars ->
  env_values_all Pv vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    forallb
      (fun binding =>
         sets_seen_give c f slot d (binding_sets_seen known binding))
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
    exact (eval_static_const_slots_give known const vars allocated Hhold
             Hconst Hvars Heval).
Qed.

Lemma value_slots_give_partial (callee : value) (args : list value)
    (args_arity : arity) :
  Pv (Value_partial_application callee args args_arity)
  <-> Pv callee ∧ Forall Pv args.
Proof.
  unfold Pv. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity.
Qed.

Lemma values_slots_give_step (s : state) :
  expr_slots_give c f slot d (state_vars s) (state_expr s) ->
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
           s (fun _ => value_slots_give_partial) Hvalid).
  - intros size tuple fields Htuple Hfields.
    destruct Hvalid as (_ & _ & Hstatics & _).
    apply (tuple_fields_forall _ (state_statics s) size tuple fields);
      [ | exact Hfields ].
    intros field v Hprim.
    refine (eval_prim_preserves Pv (state_statics s) _ [tuple] v
              value_slots_give_of_numbers value_slots_give_of_blocks
              value_slots_give_of_arrays
              (value_slots_give_in_static_blocks (state_statics s) Hstatics)
              (value_slots_give_in_static_arrays (state_statics s) Hstatics) _
              Hprim).
    intros w [<- | []]. exact Htuple.
  - intros x kind defining body v hp' rd' Hexpr Hexec. rewrite Hexpr in P.
    cbn [sets_seen] in P. rewrite sets_seen_give_app in P.
    apply andb_true_iff in P as [Pd _].
    destruct Hvalid as (Hvars & _ & Hstatics & Hheap).
    refine
      (exec_named_values_all Pv (state_statics s) (state_heap s) hp'
         (state_reads s) rd'
              (state_vars s) defining v value_slots_give_of_numbers
              value_slots_give_of_blocks value_slots_give_of_arrays
              (value_slots_give_in_static_blocks (state_statics s) Hstatics)
              (value_slots_give_in_static_arrays (state_statics s) Hstatics)
              (fun _ => eq_refl) Hheap _ _ Hexec).
    + intros p args vs _ Hvs.
      exact (proj1 (forallb_forall _ _)
               (eval_simples_slots_give (state_vars s) Hvars args vs Hvs)).
    + intros _ w Heval.
      exact (eval_named_slots_give (state_statics s) known (state_vars s)
               defining w Hhold Pd Hvars Hstatics Heval).
  - intros bindings body statics' Hexpr Heval.
    split; [ exact (fun v H => H) | ].
    rewrite Hexpr in P. cbn [sets_seen] in P.
    rewrite sets_seen_give_app, bindings_sets_seen_give in P.
    apply andb_true_iff in P as [Pbindings _].
    destruct Hvalid as (Hvars & _ & Hstatics & _).
    exact (eval_static_bindings_slots_give known (state_vars s) Hhold Hvars
             bindings (state_statics s) statics' Pbindings Hstatics Heval).
  - intros ap callee vs extra _ Hcallee Hargs Hextra.
    destruct Hvalid as (Hvars & _).
    split; [ | split ].
    + exact (eval_simple_slots_give (state_vars s) (ap_callee ap) callee Hvars
               Hcallee).
    + exact (values_slots_give_Forall vs
               (eval_simples_slots_give (state_vars s) Hvars (ap_args ap) vs
                  Hargs)).
    + exact (values_slots_give_Forall extra
               (eval_simples_slots_give (state_vars s) Hvars
                  (exn_extra_simples ap) extra Hextra)).
  - intros ac vs _ Hargs. destruct Hvalid as (Hvars & _).
    exact (values_slots_give_Forall vs
             (eval_simples_slots_give (state_vars s) Hvars (ac_args ac) vs
                Hargs)).
Qed.

(* The terms. *)
Definition terms_give (s : state) : Prop :=
  expr_slots_give c f slot d (state_vars s) (state_expr s)
  ∧ conts_slots_give c f slot d (state_conts s)
  ∧ code_slots_give c f slot d (state_statics s).

Lemma conts_slots_give_app (conts conts' : cont_env) :
  conts_slots_give c f slot d conts ->
  conts_slots_give c f slot d conts' ->
  conts_slots_give c f slot d (conts ++ conts').
Proof.
  unfold conts_slots_give.
  induction conts as [ | kc conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] H'. split; [ exact Hc | exact (IH Hrest H') ].
Qed.

Lemma resolve_slots_give (k : continuation) (conts : cont_env) :
  conts_slots_give c f slot d conts ->
  cont_slots_give c f slot d (resolve k conts).
Proof.
  unfold resolve, conts_slots_give.
  induction conts as [ | [k' cv] conts IH]; simpl; [ intros _; exact I | ].
  intros [Hc Hrest]. destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma rec_closures_slots_give (known : closure_sight) (vars : env)
    (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  sight_holds known vars ->
  forallb
    (fun '(_, h) => sets_seen_give c f slot d (handler_sets_seen known h)) all
  = true ->
  conts_slots_give c f slot d conts ->
  forallb
    (fun '(_, h) => sets_seen_give c f slot d (handler_sets_seen known h))
    handlers
  = true ->
  conts_slots_give c f slot d
    (map (fun '(k, h) => (k, Closure h vars conts all)) handlers).
Proof.
  intros Hhold Hall Hconts. unfold conts_slots_give.
  induction handlers as [ | [k h] handlers IH]; simpl;
    [ intros _; exact I | ].
  intros H. apply andb_true_iff in H as [Hh Hrest].
  split; [ | exact (IH Hrest) ].
  split; [ | exact Hconts ].
  exists known. exact (conj Hhold (conj Hh Hall)).
Qed.

Lemma with_extra_args_slots_give (cv : cont_value) (extra : list value) :
  cont_slots_give c f slot d cv ->
  cont_slots_give c f slot d (with_extra_args cv extra).
Proof. destruct extra; intros Hc; exact Hc. Qed.

Lemma eval_static_bindings_code_slots_give (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    (∀ (id : code_id) (code : code),
      Bind_code id code ∈ bindings ->
      sets_seen_give c f slot d (code_sets_seen code) = true) ->
    code_slots_give c f slot d statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    code_slots_give c f slot d statics'.
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
Lemma terms_give_over (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_slots_give c f slot d
    (Over_application remaining remaining_arity return_arity return_to
       exn_to)
  <-> cont_slots_give c f slot d return_to
      ∧ cont_slots_give c f slot d exn_to.
Proof. reflexivity. Qed.

Lemma terms_give_code (statics : static_env) (hp : heap) (rd : read_record) :
  code_slots_give c f slot d statics ->
  ∀ (called : code_id) (code : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some code ->
    cont_slots_give c f slot d return_to ->
    cont_slots_give c f slot d exn_to ->
    terms_give
      (State (code_body code) callee_vars
         [(code_return_continuation code, return_to);
          (code_exn_continuation code, exn_to)]
         statics hp rd).
Proof.
  intros S called code callee_vars return_to exn_to Hlookup Hreturn Hexn.
  unfold terms_give. simpl_state.
  refine (conj _ (conj (conj Hreturn (conj Hexn I)) S)).
  exists []. split; [ apply sight_holds_nil | ].
  pose proof (S called code Hlookup) as H.
  destruct code as [params my_closure ret exn metadata body]. exact H.
Qed.

Lemma terms_give_handler (statics : static_env) (hp : heap)
    (rd : read_record) :
  code_slots_give c f slot d statics ->
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (handler_vars : env),
    cont_slots_give c f slot d
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    bind_params params vs def_vars = Some handler_vars ->
    terms_give
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros S params metadata handler def_vars def_conts recursive vs
    handler_vars Cr Hbind.
  unfold terms_give. simpl_state. cbn [cont_slots_give] in Cr.
  destruct Cr as ((known & Hhold & Hh & Hrec) & Cdef).
  refine (conj _ (conj (conts_slots_give_app _ _ _ Cdef) S)).
  - exists (sight_without params known). split; [ | exact Hh ].
    exact (sight_holds_bind_params params known vs def_vars handler_vars
             Hhold Hbind).
  - exact (rec_closures_slots_give known def_vars def_conts recursive
             recursive Hhold Hrec Cdef Hrec).
Qed.

Lemma terms_give_step (s s' : state) :
  terms_give s -> step s = Step s' -> terms_give s'.
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
    injection Hstep as <-. unfold terms_give. simpl_state.
    refine (conj _ (conj C S)).
    exists (sight_under_let known x defining). split.
    + apply sight_holds_under_let; [ exact Hhold | ].
      intros seen Hmade.
      exact (exec_named_sight _ _ _ _ _ known vars defining seen v Hhold
               Hmade Hexec).
    + cbn [sets_seen] in P. rewrite sets_seen_give_app in P.
      apply andb_true_iff in P as [_ P]. exact P.
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold terms_give. simpl_state.
    cbn [sets_seen] in P.
    rewrite sets_seen_give_app, bindings_sets_seen_give in P.
    apply andb_true_iff in P as [Pbindings Pbody].
    refine (conj _ (conj C _)).
    + exists known. exact (conj Hhold Pbody).
    + apply (eval_static_bindings_code_slots_give vars bindings statics
               statics'); [ | exact S | exact Heval ].
      intros id code Hin. rewrite forallb_forall in Pbindings.
      exact (Pbindings _ Hin).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold terms_give. simpl_state.
    cbn [sets_seen] in P. rewrite sets_seen_give_app in P.
    apply andb_true_iff in P as [Hh Pbody].
    refine (conj _ (conj (conj _ C) S)).
    + exists known. exact (conj Hhold Pbody).
    + cbn [snd cont_slots_give]. split; [ | exact C ].
      exists known. exact (conj Hhold (conj Hh eq_refl)).
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold terms_give. simpl_state.
    cbn [sets_seen] in P.
    rewrite sets_seen_give_app, handlers_sets_seen_give in P.
    apply andb_true_iff in P as [Phandlers Pbody].
    refine (conj _ (conj (conts_slots_give_app _ _ _ C) S)).
    + exists known. exact (conj Hhold Pbody).
    + exact (rec_closures_slots_give known vars conts handlers handlers Hhold
               Phandlers C Phandlers).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    refine
      (call_function_lands_bound statics _ rd_ (cont_slots_give c f slot d)
         terms_give (fun cv extra H => H) terms_give_over
         (terms_give_code statics _ rd_ S)
         (terms_give_handler statics _ rd_ S) _ _ _ _ _ _ _ s'
         (resolve_slots_give _ conts C) _ Hstep).
    apply with_extra_args_slots_give, resolve_slots_give, C.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    exact
      (apply_cont_value_lands_bound statics _ rd_
         (cont_slots_give c f slot d) terms_give (fun cv extra H => H)
         terms_give_over (terms_give_code statics _ rd_ S)
         (terms_give_handler statics _ rd_ S) _ vs s'
         (resolve_slots_give _ conts C) Hstep).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    injection Hstep as <-. unfold terms_give. simpl_state.
    refine (conj _ (conj C S)).
    exists known. split; [ exact Hhold | reflexivity ].
  - discriminate.
Qed.

Lemma state_slots_give_terms (s : state) :
  state_slots_give c f slot d s
  <-> terms_give s
      ∧ state_values_all (fun _ v => Pv v) (fun _ allocated => Qv allocated)
          s.
Proof. unfold state_slots_give, terms_give, Pv, Qv. tauto. Qed.

Lemma state_slots_give_kept : kept_by_step (state_slots_give c f slot d).
Proof.
  intros s s' Hinv Hstep.
  apply state_slots_give_terms in Hinv as [Hterms Hvalues].
  apply state_slots_give_terms. split.
  - exact (terms_give_step s s' Hterms Hstep).
  - pose proof (values_slots_give_step s (proj1 Hterms) Hvalues) as H.
    rewrite Hstep in H. exact H.
Qed.

Lemma closure_slot_gives_aux (s : state) (x : variable) (v : value)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  state_slots_give c f slot d s ->
  lookup x (state_vars s) = Some v ->
  closure_in_set (state_statics s) v = Some (f, function_decls, slot_values) ->
  lookup f function_decls = Some c ->
  ∃ w : value,
    lookup slot slot_values = Some w ∧ value_is_closure_of d w = true.
Proof.
  intros (_ & _ & _ & Hvars & _ & Hstatics & _) Hx Hin_set Hcode.
  destruct (closure_in_set_slots_give (state_statics s) v f function_decls
              slot_values Hstatics
              (env_values_all_lookup _ (state_vars s) x v Hvars Hx) Hin_set)
    as [Hset _].
  unfold set_held_gives, has_code_at in Hset.
  rewrite Hcode, Pos.eqb_refl in Hset. simpl in Hset.
  destruct (lookup slot slot_values) as [w | ]; [ | discriminate ].
  exists w. split; [ reflexivity | exact Hset ].
Qed.

End Slots_give.

Theorem state_slots_give_from : state_slots_give_from_statement.
Proof.
  intros c f slot d e conts statics He Hconts Hvalues Hcode Hstatics.
  unfold state_slots_give, state_values_all. simpl_state.
  refine (conj _ (conj Hconts (conj Hcode _))).
  - exists []. split; [ apply sight_holds_nil | exact He ].
  - exact
      (conj (Forall_nil _)
         (conj Hvalues (conj Hstatics (heap_values_all_empty _)))).
Qed.

Theorem state_slots_give_initial : state_slots_give_initial_statement.
Proof.
  intros c f slot d e He.
  apply state_slots_give_from; [ exact He | exact I | exact I | | ].
  - intros id code Hlookup. discriminate Hlookup.
  - apply Forall_nil.
Qed.

Theorem state_slots_give_step : state_slots_give_step_statement.
Proof. intros c f slot d. exact (state_slots_give_kept c f slot d). Qed.

Theorem closure_slot_gives : closure_slot_gives_statement.
Proof.
  intros c f slot d s x v function_decls slot_values.
  apply closure_slot_gives_aux.
Qed.

Theorem closure_value_of : closure_value_of_statement.
Proof.
  intros statics d w H. unfold value_is_closure_of in H.
  destruct w as [k | code slots | at_slot decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    simpl in H; try discriminate.
  - apply Pos.eqb_eq in H. subst d. exists slots. reflexivity.
  - destruct (lookup at_slot decls) as [code | ] eqn:Hslot;
      simpl in H; [ | discriminate ].
    unfold seen_code in H. simpl in H. rewrite Hslot in H.
    apply Pos.eqb_eq in H. subst d. exists slots. simpl. rewrite Hslot.
    reflexivity.
Qed.
