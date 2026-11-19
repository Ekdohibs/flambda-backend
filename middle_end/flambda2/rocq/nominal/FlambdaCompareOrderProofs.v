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

(* Proofs of the statements of FlambdaCompareOrderSpec.v. There is no
   statement here, and nothing here needs review.

   That the comparison is sound is proved as FlambdaCompareProofs.v
   proves it of [expr_eqb], with a relation that has a first side and
   a second. Two states are related ([state_ordered]) when
   [expr_says_at_least] accepts their expressions, the first being the
   one that may state more, under environments that give paired names
   the same things. The values of the two runs are the same. The
   continuation closures and the code are not: their bound names
   differ, and the first may state more of its parameters; so those
   are related, the same way round ([cont_ordered],
   [statics_ordered]). A step of the first state that is defined is
   matched by a step of the second ([ordered_step]), which gives the
   refinement (FlambdaSemanticsProofs.simulation_refines): where the
   first state is undefined nothing is asked of the second.

   The two proofs part only where the machine reads what a binder
   states, at the entry of a handler and of code. There the values
   are the same on the two sides, and those that pass the test of
   what the first states pass the test of what the second states
   (FlambdaSubkindOrderProofs.values_of_subkinds_weaken); if they
   fail the first, the first state is undefined. The lemmas of
   FlambdaCompareProofs.v that do not mention how two expressions are
   compared are used as they are. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaSubkindSpec FlambdaSubkindProofs FlambdaSubkindOrderSpec
  FlambdaSubkindOrderProofs FlambdaNominalSpec FlambdaCompareSpec
  FlambdaCompareProofs FlambdaCompareOrderSpec.
Import ListNotations.

(* ======================================================================== *)
(* What [expr_eqb] accepts                                                  *)
(* ======================================================================== *)

Lemma handler_metadata_eqb_says_at_least (m m' : handler_metadata) :
  handler_metadata_eqb m m' = true ->
  handler_metadata_says_at_least m m' = true.
Proof.
  unfold handler_metadata_eqb, handler_metadata_says_at_least. intros H.
  apply andb_true_iff in H as [H Hexn].
  apply andb_true_iff in H as [Hkinds Hsubkinds].
  apply param_subkinds_eqb_eq in Hsubkinds.
  rewrite Hkinds, Hexn, Hsubkinds,
    (params_say_at_least_refl (hm_params_subkinds m')).
  reflexivity.
Qed.

Lemma code_metadata_eqb_says_at_least (m m' : code_metadata) :
  code_metadata_eqb m m' = true -> code_metadata_says_at_least m m' = true.
Proof.
  unfold code_metadata_eqb, code_metadata_says_at_least. intros H.
  apply andb_true_iff in H as [H Hstub].
  apply andb_true_iff in H as [H Htupled].
  apply andb_true_iff in H as [H Hresults].
  apply andb_true_iff in H as [H Hresult].
  apply andb_true_iff in H as [Hparams Hsubkinds].
  apply param_subkinds_eqb_eq in Hsubkinds.
  apply param_subkinds_eqb_eq in Hresults.
  rewrite Hparams, Hresult, Htupled, Hstub, Hsubkinds, Hresults,
    (params_say_at_least_refl (cm_params_subkinds m')),
    (params_say_at_least_refl (cm_result_subkinds m')).
  reflexivity.
Qed.

Fixpoint expr_eqb_says_at_least_at (e : expr) :
  ∀ (vm km : name_map) (e' : expr),
    expr_eqb vm km e e' = true -> expr_says_at_least vm km e e' = true.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros vm km
      [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
      | ap' | ac' | scrutinee' arms' | ];
    simpl; try discriminate; try (intros Heq; exact Heq); intros Heq.
  - apply andb_true_iff in Heq as [Hd Hbody].
    rewrite Hd, (expr_eqb_says_at_least_at body _ _ _ Hbody). reflexivity.
  - apply andb_true_iff in Heq as [Hbindings Hbody].
    rewrite (expr_eqb_says_at_least_at body _ _ _ Hbody), andb_true_r.
    clear Hbody. revert bindings' Hbindings.
    induction bindings as [ | binding bindings IH];
      intros [ | binding' bindings']; simpl; try discriminate;
      [ reflexivity | ].
    intros Heq. apply andb_true_iff in Heq as [Hbinding Hrest].
    rewrite (IH _ Hrest), andb_true_r.
    destruct binding as [id c | id | s const],
      binding' as [id' c' | id' | s' const'];
      simpl in Hbinding |- *; try discriminate; try exact Hbinding.
    apply andb_true_iff in Hbinding as [Hid Hc]. rewrite Hid. simpl.
    destruct c as [params my_closure ret exn metadata code_body],
      c' as [params' my_closure' ret' exn' metadata' code_body'].
    simpl in Hc |- *.
    destruct (bind_names params params' [(my_closure, my_closure')])
      as [vm0 | ]; [ | discriminate ].
    apply andb_true_iff in Hc as [Hmetadata Hc].
    rewrite (code_metadata_eqb_says_at_least _ _ Hmetadata),
      (expr_eqb_says_at_least_at code_body _ _ _ Hc).
    reflexivity.
  - apply andb_true_iff in Heq as [Hh Hbody].
    rewrite (expr_eqb_says_at_least_at body _ _ _ Hbody), andb_true_r.
    destruct h as [params metadata handler],
      h' as [params' metadata' handler'].
    simpl in Hh |- *.
    destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
    apply andb_true_iff in Hh as [Hmetadata Hh].
    rewrite (handler_metadata_eqb_says_at_least _ _ Hmetadata),
      (expr_eqb_says_at_least_at handler _ _ _ Hh).
    reflexivity.
  - destruct (combine_names (map fst handlers) (map fst handlers') km)
      as [km' | ]; [ | discriminate ].
    apply andb_true_iff in Heq as [Hhandlers Hbody].
    rewrite (expr_eqb_says_at_least_at body _ _ _ Hbody), andb_true_r.
    clear Hbody. revert handlers' Hhandlers.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      intros [ | [k' [params' metadata' handler']] handlers']; simpl;
      try discriminate; [ reflexivity | ].
    intros Heq. apply andb_true_iff in Heq as [Hh Hrest].
    destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
    apply andb_true_iff in Hh as [Hmetadata Hh].
    rewrite (handler_metadata_eqb_says_at_least _ _ Hmetadata),
      (expr_eqb_says_at_least_at handler _ _ _ Hh), (IH _ Hrest).
    reflexivity.
Qed.

Theorem expr_eqb_says_at_least : expr_eqb_says_at_least_statement.
Proof. intros vm km e e'. apply expr_eqb_says_at_least_at. Qed.

Lemma handler_eqb_says_at_least (vm km : name_map)
    (h h' : continuation_handler) :
  handler_eqb vm km h h' = true -> handler_says_at_least vm km h h' = true.
Proof.
  destruct h as [params metadata handler],
    h' as [params' metadata' handler'].
  simpl. destruct (bind_names params params' vm) as [vm' | ];
    [ | discriminate ].
  intros Hh. apply andb_true_iff in Hh as [Hmetadata Hh].
  rewrite (handler_metadata_eqb_says_at_least _ _ Hmetadata),
    (expr_eqb_says_at_least_at handler _ _ _ Hh).
  reflexivity.
Qed.

Lemma code_eqb_says_at_least (c c' : code) :
  code_eqb c c' = true -> code_says_at_least c c' = true.
Proof.
  destruct c as [params my_closure ret exn metadata body],
    c' as [params' my_closure' ret' exn' metadata' body'].
  simpl.
  destruct (bind_names params params' [(my_closure, my_closure')])
    as [vm | ]; [ | discriminate ].
  intros Hc. apply andb_true_iff in Hc as [Hmetadata Hc].
  rewrite (code_metadata_eqb_says_at_least _ _ Hmetadata),
    (expr_eqb_says_at_least_at body _ _ _ Hc).
  reflexivity.
Qed.

(* ======================================================================== *)
(* What the comparison of two metadata gives                                *)
(* ======================================================================== *)

Lemma handler_metadata_says_at_least_inv (m m' : handler_metadata) :
  handler_metadata_says_at_least m m' = true ->
  hm_params_kinds m' = hm_params_kinds m
  ∧ hm_is_exn_handler m' = hm_is_exn_handler m
  ∧ params_say_at_least (hm_params_subkinds m) (hm_params_subkinds m')
    = true.
Proof.
  unfold handler_metadata_says_at_least. intros H.
  apply andb_true_iff in H as [H Hexn].
  apply andb_true_iff in H as [Hkinds Hsay].
  apply kinds_eqb_eq in Hkinds. apply Bool.eqb_prop in Hexn.
  repeat split; [ symmetry; exact Hkinds | symmetry; exact Hexn | exact Hsay ].
Qed.

Lemma code_metadata_says_at_least_inv (m m' : code_metadata) :
  code_metadata_says_at_least m m' = true ->
  cm_params_arity m' = cm_params_arity m
  ∧ cm_result_arity m' = cm_result_arity m
  ∧ cm_is_tupled m' = cm_is_tupled m
  ∧ cm_is_stub m' = cm_is_stub m
  ∧ params_say_at_least (cm_params_subkinds m) (cm_params_subkinds m')
    = true.
Proof.
  unfold code_metadata_says_at_least. intros H.
  apply andb_true_iff in H as [H Hstub].
  apply andb_true_iff in H as [H Htupled].
  apply andb_true_iff in H as [H _].
  apply andb_true_iff in H as [H Hresult].
  apply andb_true_iff in H as [Hparams Hsay].
  apply arity_eqb_eq in Hparams. apply arity_eqb_eq in Hresult.
  apply Bool.eqb_prop in Htupled. apply Bool.eqb_prop in Hstub.
  repeat split; try (symmetry; assumption). exact Hsay.
Qed.


(* ======================================================================== *)
(* Static environments                                                      *)
(* ======================================================================== *)

(* The same statically allocated values and, for each code id, two
   pieces of code that the comparison accepts: the first static
   environment is that of the state that may state more. *)
Definition statics_ordered (statics statics' : static_env) : Prop :=
  Forall2
    (fun ic ic' =>
       fst ic = fst ic' ∧ code_says_at_least (snd ic) (snd ic') = true)
    (static_code statics) (static_code statics')
  ∧ static_values statics = static_values statics'.

Lemma statics_ordered_same (statics : static_env) :
  statics_ordered statics statics.
Proof.
  split; [ | reflexivity ].
  induction (static_code statics) as [ | ic code IH];
    [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | apply code_eqb_says_at_least, code_eqb_refl ].
Qed.

Lemma lookup_code_ordered (id : code_id) (statics statics' : static_env) :
  statics_ordered statics statics' ->
  match lookup id (static_code statics), lookup id (static_code statics') with
  | Some c, Some c' => code_says_at_least c c' = true
  | None, None => True
  | _, _ => False
  end.
Proof.
  intros [Hcode _].
  induction Hcode as [ | [id1 c] [id1' c'] code code' [Hid Hc] _ IH]; simpl;
    [ exact I | ].
  simpl in Hid. subst id1'. destruct (Pos.eqb id id1); [ exact Hc | exact IH ].
Qed.

Lemma closure_of_ordered (statics statics' : static_env) (v : value) :
  statics_ordered statics statics' ->
  closure_of statics v = closure_of statics' v.
Proof.
  intros [_ Hvalues]. unfold closure_of. rewrite Hvalues. reflexivity.
Qed.

Lemma project_value_slot_ordered (statics statics' : static_env)
    (project_from : function_slot) (slot : value_slot) (v : value) :
  statics_ordered statics statics' ->
  project_value_slot statics project_from slot v
  = project_value_slot statics' project_from slot v.
Proof.
  intros [_ Hvalues]. unfold project_value_slot, closure_in_set.
  rewrite Hvalues. reflexivity.
Qed.

Lemma project_function_slot_ordered (statics statics' : static_env)
    (move_from move_to : function_slot) (v : value) :
  statics_ordered statics statics' ->
  project_function_slot statics move_from move_to v
  = project_function_slot statics' move_from move_to v.
Proof.
  intros [_ Hvalues]. unfold project_function_slot. rewrite Hvalues.
  reflexivity.
Qed.

(* What a [let] binds is compared as [expr_eqb] compares it, and reads
   of a static environment its values alone: so it comes to the same
   under two static environments that are related. *)
Lemma eval_named_ordered (vm : name_map) (vars vars' : env)
    (statics statics' : static_env) (d d' : named) :
  envs_match (fun x => var_free_in_named x d) vm vars vars' ->
  statics_ordered statics statics' ->
  named_eqb vm d d' = true ->
  eval_named statics vars d = eval_named statics' vars' d'.
Proof.
  intros Hmatch Hstatics.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure],
    d' as [s' | p' args' | code' slots' | fslot' decls' slots'
          | slot' closure' | pfrom' slot' closure'
          | move_from' move_to' closure'];
    simpl; try discriminate; intros Heq.
  - exact (eval_simple_match vm vars vars' s s' Hmatch Heq).
  - apply andb_true_iff in Heq as [Hp Hargs]. apply prim_eqb_sound in Hp.
    subst p'.
    rewrite (eval_simples_match vm vars vars' args Hmatch args' Hargs).
    destruct (eval_simples vars' args') as [vs | ]; simpl; [ | reflexivity ].
    apply eval_prim_same_static_values, Hstatics.
  - apply andb_true_iff in Heq as [Hcode Hslots]. apply Pos.eqb_eq in Hcode.
    subst code'.
    rewrite (eval_value_slots_match vm vars vars' slots Hmatch slots' Hslots).
    reflexivity.
  - apply andb_true_iff in Heq as [Heq Hslots].
    apply andb_true_iff in Heq as [Hfslot Hdecls].
    apply Pos.eqb_eq in Hfslot. apply names_eqb_eq in Hdecls.
    subst fslot' decls'.
    rewrite (eval_value_slots_match vm vars vars' slots Hmatch slots' Hslots).
    reflexivity.
  - apply andb_true_iff in Heq as [Hslot Hclosure].
    apply Pos.eqb_eq in Hslot. subst slot'.
    rewrite (eval_simple_match vm vars vars' closure closure' Hmatch Hclosure).
    destruct (eval_simple vars' closure') as [v | ]; simpl; [ | reflexivity ].
    rewrite (closure_of_ordered statics statics' v Hstatics). reflexivity.
  - apply andb_true_iff in Heq as [Heq Hclosure].
    apply andb_true_iff in Heq as [Hfrom Hslot].
    apply Pos.eqb_eq in Hfrom. apply Pos.eqb_eq in Hslot.
    subst pfrom' slot'.
    rewrite (eval_simple_match vm vars vars' closure closure' Hmatch Hclosure).
    destruct (eval_simple vars' closure') as [v | ]; simpl; [ | reflexivity ].
    apply project_value_slot_ordered, Hstatics.
  - apply andb_true_iff in Heq as [Heq Hclosure].
    apply andb_true_iff in Heq as [Hfrom Hto].
    apply Pos.eqb_eq in Hfrom. apply Pos.eqb_eq in Hto.
    subst move_from' move_to'.
    rewrite (eval_simple_match vm vars vars' closure closure' Hmatch Hclosure).
    destruct (eval_simple vars' closure') as [v | ]; simpl; [ | reflexivity ].
    apply project_function_slot_ordered, Hstatics.
Qed.

(* And it does the same on a heap. *)
Lemma exec_named_ordered (vm : name_map) (vars vars' : env)
    (statics statics' : static_env) (hp : heap) (rd : read_record)
      (d d' : named) :
  envs_match (fun x => var_free_in_named x d) vm vars vars' ->
  statics_ordered statics statics' ->
  named_eqb vm d d' = true ->
  exec_named statics hp rd vars d = exec_named statics' hp rd vars' d'.
Proof.
  intros Hmatch Hstatics Heq. symmetry. apply exec_named_agrees.
  - symmetry.
    exact (eval_named_ordered vm vars vars' statics statics' d d' Hmatch
             Hstatics Heq).
  - intros p args ->.
    destruct d' as [s' | p' args' | code' slots' | fslot' decls' slots'
          | slot' closure' | pfrom' slot' closure'
          | move_from' move_to' closure'];
      simpl in Heq; try discriminate.
    apply andb_true_iff in Heq as [Hp Hargs]. apply prim_eqb_sound in Hp.
    subst p'.
    exists args'. split; [ reflexivity | split ].
    + symmetry.
      exact (eval_simples_match vm vars vars' args Hmatch args' Hargs).
    + intros vs s _ _. rewrite (proj2 Hstatics). reflexivity.
  - intros p args ->.
    destruct d as [s | p0 args0 | | | | | ]; simpl in Heq; try discriminate.
    eauto.
Qed.

(* Two groups of bindings that the comparison accepts, added to static
   environments that are related: both fail, or the results are
   related. *)
Lemma eval_static_bindings_ordered (vm : name_map) (vars vars' : env)
    (bindings : list static_binding) :
  envs_match (fun x => Exists (var_free_in_static_binding x) bindings) vm vars
    vars' ->
  ∀ (bindings' : list static_binding) (statics statics' : static_env),
    forall2b (binding_says_at_least vm) bindings bindings' = true ->
    statics_ordered statics statics' ->
    match
      eval_static_bindings vars bindings statics,
      eval_static_bindings vars' bindings' statics'
    with
    | Some extended, Some extended' => statics_ordered extended extended'
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction bindings as [ | binding bindings IH];
    intros Hmatch [ | binding' bindings'] statics statics'; simpl;
    try discriminate; [ intros _ Hstatics; exact Hstatics | ].
  intros Heq Hstatics. pose proof Hstatics as [Hcode Hvalues].
  apply andb_true_iff in Heq as [Hbinding Hrest].
  assert (Hmatch_first :
    envs_match (fun x => var_free_in_static_binding x binding) vm vars vars')
    by (fewer_variables Hmatch; apply Exists_cons_hd; assumption).
  assert (Hmatch_rest :
    envs_match (fun x => Exists (var_free_in_static_binding x) bindings) vm
      vars vars')
    by (fewer_variables Hmatch; apply Exists_cons_tl; assumption).
  specialize (IH Hmatch_rest).
  destruct binding as [id c | id | sym const],
    binding' as [id' c' | id' | sym' const'];
    simpl in Hbinding; try discriminate.
  - apply andb_true_iff in Hbinding as [Hid Hc]. apply Pos.eqb_eq in Hid.
    subst id'.
    pose proof (lookup_code_ordered id statics statics' Hstatics) as Hid.
    destruct (lookup id (static_code statics)),
      (lookup id (static_code statics')); try contradiction; [ exact I | ].
    apply (IH bindings' _ _ Hrest). split; [ | exact Hvalues ].
    apply Forall2_cons; [ split; [ reflexivity | exact Hc ] | exact Hcode ].
  - apply (IH bindings' _ _ Hrest). split; assumption.
  - apply andb_true_iff in Hbinding as [Hsym Hconst].
    apply Pos.eqb_eq in Hsym. subst sym'. rewrite <- Hvalues.
    destruct (lookup sym (static_values statics)); [ exact I | ].
    rewrite (eval_static_const_match vm vars vars' const const' Hmatch_first
      Hconst).
    destruct (eval_static_const vars' const') as [allocated | ]; simpl;
      [ | exact I ].
    apply (IH bindings' _ _ Hrest). split; [ exact Hcode | ].
    simpl. rewrite Hvalues. reflexivity.
Qed.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

(* The handlers of a recursive group, compared one with one. *)
Definition handlers_say_at_least (vm km : name_map)
    (handlers handlers' : list (continuation * continuation_handler)) : bool :=
  forall2b (fun '(_, h) '(_, h') => handler_says_at_least vm km h h')
    handlers handlers'.

(* What two continuation names that match stand for: the same way out, the
   same continuation, or closures whose handlers the comparison accepts,
   the first being the one that may state more, under the pairs of names
   of where they were defined, with environments that match on the names
   free in the handlers: those of the handler itself and of the handlers
   defined with it, any of which it may come to run; or what is left of
   the same over-application, with continuations that are related. The
   relation is not symmetric. *)
#[local] Set Warnings "-register-all".
Inductive cont_ordered : cont_value -> cont_value -> Prop :=
  | Same_ordered (c : cont_value) : cont_ordered c c
  | Closure_ordered (vm km km' : name_map) (h h' : continuation_handler)
      (vars vars' : env) (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    combine_names (map fst recursive) (map fst recursive') km = Some km' ->
    handler_says_at_least vm km' h h' = true ->
    handlers_say_at_least vm km' recursive recursive' = true ->
    envs_match
      (fun x =>
         var_free_in_handler x h
         ∨ any_handler (var_free_in_handler x) recursive)
      vm vars vars' ->
    (∀ k k' : continuation,
      (cont_free_in_handler k h
       ∨ any_handler (cont_free_in_handler k) recursive)
      ∧ lookup k recursive = None ->
      names_match km k k' = true ->
      cont_ordered (resolve k conts) (resolve k' conts')) ->
    cont_ordered (Closure h vars conts recursive)
      (Closure h' vars' conts' recursive')
  | Extra_args_ordered (c c' : cont_value) (extra : list value) :
    cont_ordered c c' ->
    cont_ordered (Extra_args c extra) (Extra_args c' extra)
  | Over_application_ordered (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_ordered return_to return_to' ->
    cont_ordered exn_to exn_to' ->
    cont_ordered
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Lemma with_extra_args_ordered (c c' : cont_value) (extra : list value) :
  cont_ordered c c' ->
  cont_ordered (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl;
    [ exact Hc | apply Extra_args_ordered, Hc ].
Qed.

(* As [envs_match]: the names that matter are the free continuations of
   what is about to run. *)
Definition conts_ordered (relevant : continuation -> Prop) (km : name_map)
    (conts conts' : cont_env) : Prop :=
  ∀ k k' : continuation,
    relevant k -> names_match km k k' = true ->
    cont_ordered (resolve k conts) (resolve k' conts').

Lemma conts_ordered_same (relevant : continuation -> Prop) (conts : cont_env) :
  conts_ordered relevant [] conts conts.
Proof.
  intros k k' _ Hk. apply names_match_nil in Hk. subst k'. apply Same_ordered.
Qed.

Lemma conts_ordered_weaken (relevant fewer : continuation -> Prop)
    (km : name_map) (conts conts' : cont_env) :
  (∀ k : continuation, fewer k -> relevant k) ->
  conts_ordered relevant km conts conts' ->
  conts_ordered fewer km conts conts'.
Proof. intros Hfewer Hmatch k k' Hk. apply Hmatch, Hfewer, Hk. Qed.

Ltac fewer_continuations H :=
  refine (conts_ordered_weaken _ _ _ _ _ _ H); intros ? ?.

Lemma conts_ordered_cons (relevant inner : continuation -> Prop)
    (km : name_map) (conts conts' : cont_env) (a a' : continuation)
    (c c' : cont_value) :
  (∀ k : continuation, inner k -> k = a ∨ relevant k) ->
  conts_ordered relevant km conts conts' ->
  cont_ordered c c' ->
  conts_ordered inner ((a, a') :: km) ((a, c) :: conts) ((a', c') :: conts').
Proof.
  intros Hinner Hmatch Hc k k' Hk. unfold resolve. simpl.
  destruct (Pos.eqb_spec k a) as [-> | Hne].
  - intros ->. exact Hc.
  - destruct (Pos.eqb k' a'); [ discriminate | ]. apply Hmatch.
    destruct (Hinner k Hk); [ contradiction | assumption ].
Qed.

(* The environments in which the handlers of two recursive groups run,
   where the comparison accepts the groups. [all] and [all'] are the
   whole groups, kept apart from the handlers still to be gone through.
   The continuations that matter outside the group are [relevant]; among
   them are those free in the group. *)
Lemma rec_conts_ordered_aux (relevant : continuation -> Prop)
    (vm km km' : name_map) (vars vars' : env) (conts conts' : cont_env)
    (all all' : list (continuation * continuation_handler)) :
  combine_names (map fst all) (map fst all') km = Some km' ->
  handlers_say_at_least vm km' all all' = true ->
  envs_match (fun x => any_handler (var_free_in_handler x) all) vm vars vars'
  ->
  (∀ k : continuation,
    any_handler (cont_free_in_handler k) all -> lookup k all = None ->
    relevant k) ->
  conts_ordered relevant km conts conts' ->
  ∀ (handlers handlers' : list (continuation * continuation_handler))
    (inner : continuation -> Prop),
    (∀ (k : continuation) (h : continuation_handler),
      (k, h) ∈ handlers -> (k, h) ∈ all) ->
    (∀ k : continuation, inner k -> k ∈ map fst handlers ∨ relevant k) ->
    handlers_say_at_least vm km' handlers handlers' = true ->
    conts_ordered inner (combine (map fst handlers) (map fst handlers') ++ km)
      (map (fun '(k, h) => (k, Closure h vars conts all)) handlers ++ conts)
      (map (fun '(k, h) => (k, Closure h vars' conts' all')) handlers'
       ++ conts').
Proof.
  intros Hnames Hall Hvars Hgroup Hconts. unfold handlers_say_at_least.
  induction handlers as [ | [k h] handlers IH];
    intros [ | [k' h'] handlers'] inner Hin Hinner; simpl; try discriminate.
  - intros _. apply (conts_ordered_weaken relevant); [ | exact Hconts ].
    intros j Hj. destruct (Hinner j Hj) as [[] | Hrelevant]. exact Hrelevant.
  - intros Heq. apply andb_true_iff in Heq as [Hh Hrest].
    apply (conts_ordered_cons (fun j => j ∈ map fst handlers ∨ relevant j)).
    + intros j Hj. destruct (Hinner j Hj) as [[<- | Hbound] | Hrelevant]; auto.
    + apply (IH handlers'); [ | auto | exact Hrest ].
      intros j hj Hj. apply Hin. right. exact Hj.
    + pose proof (Hin k h (or_introl eq_refl)) as Hh_in.
      apply (Closure_ordered vm km km' h h' vars vars' conts conts' all all'
               Hnames Hh Hall).
      * fewer_variables Hvars.
        match goal with
        | Hx : _ ∨ _ |- _ =>
          destruct Hx as [Hx | Hx];
            [ exact (any_handler_in _ k h all Hh_in Hx) | exact Hx ]
        end.
      * intros j j' [Hj Hunbound] Hmatch. apply Hconts; [ | exact Hmatch ].
        apply Hgroup; [ | exact Hunbound ].
        destruct Hj as [Hj | Hj];
          [ exact (any_handler_in _ k h all Hh_in Hj) | exact Hj ].
Qed.

Lemma rec_conts_ordered (relevant inner : continuation -> Prop)
    (vm km km' : name_map) (vars vars' : env) (conts conts' : cont_env)
    (handlers handlers' : list (continuation * continuation_handler)) :
  combine_names (map fst handlers) (map fst handlers') km = Some km' ->
  handlers_say_at_least vm km' handlers handlers' = true ->
  envs_match (fun x => any_handler (var_free_in_handler x) handlers) vm vars
    vars' ->
  (∀ k : continuation,
    any_handler (cont_free_in_handler k) handlers ->
    lookup k handlers = None ->
    relevant k) ->
  (∀ k : continuation, inner k -> k ∈ map fst handlers ∨ relevant k) ->
  conts_ordered relevant km conts conts' ->
  conts_ordered inner km' (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hnames Hhandlers Hvars Hgroup Hinner Hconts.
  pose proof (rec_conts_ordered_aux relevant vm km km' vars vars' conts conts'
    handlers handlers' Hnames Hhandlers Hvars Hgroup Hconts handlers
    handlers' inner (fun k h Hin => Hin) Hinner Hhandlers)
    as Hmatch.
  assert (Hkm' :
    km' = combine (map fst handlers) (map fst handlers') ++ km).
  { unfold combine_names in Hnames.
    match type of Hnames with
    | context [if ?same_length then _ else _] => destruct same_length
    end;
      cbv iota in Hnames; [ | discriminate ].
    injection Hnames as Hkm'. symmetry. exact Hkm'. }
  rewrite Hkm'. exact Hmatch.
Qed.

(* ======================================================================== *)
(* States                                                                   *)
(* ======================================================================== *)

(* Two states are related when the comparison accepts their expressions,
   the first being the one that may state more, under environments that
   are related; the heap and the record of reads are the same. *)
Inductive state_ordered : state -> state -> Prop :=
  | State_ordered (vm km : name_map) (e e' : expr) (vars vars' : env)
      (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
        (rd : read_record) :
    expr_says_at_least vm km e e' = true ->
    envs_match (fun x => var_free_in_expr x e) vm vars vars' ->
    conts_ordered (fun k => cont_free_in_expr k e) km conts conts' ->
    statics_ordered statics statics' ->
    state_ordered (State e vars conts statics hp rd)
      (State e' vars' conts' statics' hp rd).

(* A continuation closure is related to itself: [expr_eqb] is reflexive,
   and the comparison accepts what [expr_eqb] accepts. *)
Lemma closure_ordered_itself (h : continuation_handler) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler)) :
  ∃ km' : name_map,
    combine_names (map fst recursive) (map fst recursive) [] = Some km'
    ∧ handler_says_at_least [] km' h h = true
    ∧ handlers_say_at_least [] km' recursive recursive = true.
Proof.
  destruct (combine_names_refl (map fst recursive) [] reflexive_map_nil)
    as (km' & Hnames & Hkm').
  exists km'. split; [ exact Hnames | ].
  split;
    [ apply handler_eqb_says_at_least, handler_eqb_refl;
        [ apply reflexive_map_nil | exact Hkm' ]
    | ].
  unfold handlers_say_at_least. apply forall2b_refl. intros [k h0].
  apply handler_eqb_says_at_least, handler_eqb_refl;
    [ apply reflexive_map_nil | exact Hkm' ].
Qed.

(* Two expressions that the comparison accepts agree on whether they
   bind something static. *)
Fixpoint expr_says_at_least_binds_statics (e : expr) :
  ∀ (vm km : name_map) (e' : expr),
    expr_says_at_least vm km e e' = true -> binds_statics e = binds_statics e'.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros vm km
      [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
      | ap' | ac' | scrutinee' arms' | ];
    simpl; try discriminate; try reflexivity; intros Heq.
  - apply andb_true_iff in Heq as [_ Hbody].
    exact (expr_says_at_least_binds_statics body _ _ _ Hbody).
  - apply andb_true_iff in Heq as [Hbindings Hbody].
    rewrite (expr_says_at_least_binds_statics body _ _ _ Hbody). f_equal.
    clear Hbody. revert bindings' Hbindings.
    induction bindings as [ | binding bindings IH];
      intros [ | binding' bindings']; simpl; try discriminate;
      [ reflexivity | ].
    intros Heq. apply andb_true_iff in Heq as [Hbinding Hrest].
    rewrite (IH _ Hrest). f_equal.
    destruct binding, binding'; simpl in Hbinding; try discriminate;
      reflexivity.
  - apply andb_true_iff in Heq as [Hh Hbody].
    destruct h as [params metadata handler],
      h' as [params' metadata' handler'].
    simpl in Hh |- *.
    destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
    apply andb_true_iff in Hh as [_ Hh].
    rewrite (expr_says_at_least_binds_statics handler _ _ _ Hh),
      (expr_says_at_least_binds_statics body _ _ _ Hbody).
    reflexivity.
  - destruct (combine_names (map fst handlers) (map fst handlers') km)
      as [km' | ]; [ | discriminate ].
    apply andb_true_iff in Heq as [Hhandlers Hbody].
    rewrite (expr_says_at_least_binds_statics body _ _ _ Hbody). f_equal.
    clear Hbody. revert handlers' Hhandlers.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      intros [ | [k' [params' metadata' handler']] handlers']; simpl;
      try discriminate; [ reflexivity | ].
    intros Heq. apply andb_true_iff in Heq as [Hh Hrest].
    destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
    apply andb_true_iff in Hh as [_ Hh].
    rewrite (expr_says_at_least_binds_statics handler _ _ _ Hh), (IH _ Hrest).
    reflexivity.
Qed.

Lemma handlers_say_at_least_bind_statics (vm km : name_map)
    (handlers : list (continuation * continuation_handler)) :
  ∀ handlers' : list (continuation * continuation_handler),
    forall2b (fun '(_, h) '(_, h') => handler_says_at_least vm km h h') handlers
      handlers'
    = true ->
    handlers_bind_statics handlers = handlers_bind_statics handlers'.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH];
    intros [ | [k' [params' metadata' handler']] handlers']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hh Hrest].
  destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
  apply andb_true_iff in Hh as [_ Hh].
  rewrite (expr_says_at_least_binds_statics handler _ _ _ Hh), (IH _ Hrest).
  reflexivity.
Qed.

(* Static environments that are related bind a code id to two pieces of
   code that the comparison accepts, which have the same arities and are
   tupled or not alike. *)
Lemma codes_agree_ordered (statics statics' : static_env) (called : code_id) :
  statics_ordered statics statics' ->
  codes_agree (fun c c' => code_says_at_least c c' = true) statics statics'
    called.
Proof.
  intros Hstatics. unfold codes_agree.
  pose proof (lookup_code_ordered called statics statics' Hstatics) as Hc.
  destruct (lookup called (static_code statics))
    as [[params my_closure ret exn metadata body] | ],
    (lookup called (static_code statics'))
    as [[params' my_closure' ret' exn' metadata' body'] | ];
    try contradiction; [ | exact I ].
  split; [ exact Hc | ]. simpl in Hc |- *.
  destruct (bind_names params params' [(my_closure, my_closure')]);
    [ | discriminate ].
  apply andb_true_iff in Hc as [Hmetadata _].
  destruct (code_metadata_says_at_least_inv _ _ Hmetadata)
    as (Hparams & Hresult & Htupled & _).
  repeat split; assumption.
Qed.

(* The same full call of two pieces of code that the comparison accepts,
   returning to continuations that are related. The arguments are the
   same values. If they are not of what the first piece states of its
   parameters, the first call is undefined; if they are, they are of
   what the second states, which is no more. *)
Lemma call_code_ordered (closure : value) (c c' : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  code_says_at_least c c' = true ->
  statics_ordered statics statics' ->
  cont_ordered return_to return_to' ->
  cont_ordered exn_to exn_to' ->
  result_simulated state_ordered
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c' vs args_arity return_arity return_to' exn_to'
       statics' hp rd).
Proof.
  intros Hc Hstatics Hreturn Hexn.
  destruct c as [params my_closure ret exn metadata body],
    c' as [params' my_closure' ret' exn' metadata' body'].
  unfold call_code. simpl in Hc |- *.
  destruct (bind_names params params' [(my_closure, my_closure')])
    as [vm | ] eqn:Hbind; [ | discriminate ].
  apply andb_true_iff in Hc as [Hmetadata Hc].
  destruct (code_metadata_says_at_least_inv _ _ Hmetadata)
    as (Hparams & Hresult & _ & _ & Hsay).
  rewrite <- (expr_says_at_least_binds_statics body _ _ _ Hc), Hparams,
    Hresult.
  destruct (binds_statics body); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (cm_params_arity metadata))
        && arity_eqb return_arity (cm_result_arity metadata)));
    [ exact I | ].
  rewrite <- (values_of_subkinds_same_values statics statics' vs
    (cm_params_subkinds metadata') (proj2 Hstatics)).
  destruct (values_of_subkinds statics vs (cm_params_subkinds metadata))
    eqn:Hsubkinds; simpl; [ | exact I ].
  rewrite (values_of_subkinds_weaken statics vs _ _ Hsay Hsubkinds). simpl.
  pose proof (bind_params_match params params' vs _ vm
    [(my_closure, closure)] [(my_closure', closure)] (fun _ => True)
    (fun x => var_free_in_expr x body) (fun x _ => or_intror I) Hbind
    (envs_match_cons (fun _ => True) (fun _ => True) [] [] [] my_closure
       my_closure' closure (fun x _ => or_intror I)
       (envs_match_same _ [])))
    as Hbound.
  destruct (bind_params params vs [(my_closure, closure)]),
    (bind_params params' vs [(my_closure', closure)]);
    try contradiction; simpl; [ | exact I ].
  apply (State_ordered vm [(ret, ret'); (exn, exn')]);
    [ exact Hc | exact Hbound | | exact Hstatics ].
  apply (conts_ordered_cons (fun _ => True)); [ auto | | exact Hreturn ].
  apply (conts_ordered_cons (fun _ => True));
    [ auto | apply conts_ordered_same | exact Hexn ].
Qed.

(* Calls of the same shape, given that what is done with a partial
   application is related. *)
Lemma run_call_ordered (return_partial return_partial' : value -> step_result)
    (shape shape' : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  shapes_related (fun c c' => code_says_at_least c c' = true) shape shape' ->
  statics_ordered statics statics' ->
  cont_ordered return_to return_to' ->
  cont_ordered exn_to exn_to' ->
  (∀ partial : value,
    result_simulated state_ordered (return_partial partial)
      (return_partial' partial)) ->
  result_simulated state_ordered
    (run_call return_partial shape return_arity return_to exn_to statics hp rd)
    (run_call return_partial' shape' return_arity return_to' exn_to'
       statics' hp rd).
Proof.
  intros Hshapes Hstatics Hreturn Hexn Hpartial.
  destruct Hshapes
    as [ | closure c c' vs args_arity Hc
       | closure c c' vs remaining args_arity remaining_arity Hc | partial];
    simpl.
  - exact I.
  - apply call_code_ordered; assumption.
  - assert (Hresult : code_result_arity c' = code_result_arity c).
    { destruct c as [params my_closure ret exn metadata body],
        c' as [params' my_closure' ret' exn' metadata' body'].
      simpl in Hc |- *.
      destruct (bind_names params params' [(my_closure, my_closure')]);
        [ | discriminate ].
      apply andb_true_iff in Hc as [Hmetadata _].
      exact (proj1 (proj2 (code_metadata_says_at_least_inv _ _ Hmetadata))). }
    rewrite Hresult.
    apply call_code_ordered; try assumption.
    apply Over_application_ordered; assumption.
  - apply Hpartial.
Qed.

(* Applying two continuations that are related to the same values, under
   static environments that are related. At a closure the values are
   tested against what its handler states of its parameters, as at a
   call. *)
Lemma apply_cont_value_ordered (c c' : cont_value) (vs : list value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  cont_ordered c c' ->
  statics_ordered statics statics' ->
  result_simulated state_ordered (apply_cont_value c vs statics hp rd)
    (apply_cont_value c' vs statics' hp rd).
Proof.
  intros Hc Hstatics. revert c' vs Hc.
  assert (Hclosure :
    ∀ (vs : list value) (vm km km' : name_map)
      (h h' : continuation_handler)
      (vars vars' : env) (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)),
      combine_names (map fst recursive) (map fst recursive') km = Some km' ->
      handler_says_at_least vm km' h h' = true ->
      handlers_say_at_least vm km' recursive recursive' = true ->
      envs_match
        (fun x =>
           var_free_in_handler x h
           ∨ any_handler (var_free_in_handler x) recursive)
        vm vars vars' ->
      conts_ordered
        (fun k =>
           (cont_free_in_handler k h
            ∨ any_handler (cont_free_in_handler k) recursive)
           ∧ lookup k recursive = None)
        km conts conts' ->
      result_simulated state_ordered
        (apply_cont_value (Closure h vars conts recursive) vs statics hp rd)
        (apply_cont_value (Closure h' vars' conts' recursive') vs statics' hp
           rd)).
  { intros vs vm km km' [params metadata handler]
      [params' metadata' handler'] vars vars' conts conts' recursive
      recursive' Hnames Hh Hrec Hvars Hconts.
    simpl in Hh |- *.
    destruct (bind_names params params' vm) as [vm' | ] eqn:Hbind;
      [ | discriminate ].
    apply andb_true_iff in Hh as [Hmetadata Hh].
    destruct (handler_metadata_says_at_least_inv _ _ Hmetadata)
      as (Hkinds & _ & Hsay).
    rewrite Hkinds.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    rewrite <- (values_of_subkinds_same_values statics statics' vs
      (hm_params_subkinds metadata') (proj2 Hstatics)).
    destruct (values_of_subkinds statics vs (hm_params_subkinds metadata))
      eqn:Hsubkinds; [ | exact I ].
    rewrite (values_of_subkinds_weaken statics vs _ _ Hsay Hsubkinds).
    pose proof (bind_params_match params params' vs vm vm' vars vars' _
      (fun x => var_free_in_expr x handler)
      (fun x Hx =>
         match in_dec Pos.eq_dec x params with
         | left Hin => or_introl Hin
         | right Hnotin => or_intror (or_introl (conj Hnotin Hx))
         end)
      Hbind Hvars) as Hbound.
    destruct (bind_params params vs vars), (bind_params params' vs vars');
      try contradiction; simpl; [ | exact I ].
    apply (State_ordered vm' km');
      [ exact Hh | exact Hbound | | exact Hstatics ].
    refine
      (rec_conts_ordered _ _ vm km km' vars vars' conts conts' recursive
         recursive' Hnames Hrec _ _ _ Hconts).
    - fewer_variables Hvars. right. assumption.
    - intros k Hk Hunbound. split; [ right; exact Hk | exact Hunbound ].
    - intros k Hk.
      destruct (lookup_none_or_in k recursive) as [Hunbound | Hin];
        [ right | left; exact Hin ].
      split; [ left; exact Hk | exact Hunbound ]. }
  assert (Hover :
    ∀ (vs : list value) (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value),
      (∀ (c' : cont_value) (vs' : list value),
        cont_ordered return_to c' ->
        result_simulated state_ordered
          (apply_cont_value return_to vs' statics hp rd)
          (apply_cont_value c' vs' statics' hp rd)) ->
      cont_ordered return_to return_to' ->
      cont_ordered exn_to exn_to' ->
      result_simulated state_ordered
        (apply_cont_value
           (Over_application remaining remaining_arity return_arity return_to
              exn_to)
           vs statics hp rd)
        (apply_cont_value
           (Over_application remaining remaining_arity return_arity
              return_to' exn_to')
           vs statics' hp rd)).
  { intros vs remaining remaining_arity return_arity return_to return_to'
      exn_to exn_to' IHreturn Hreturn Hexn.
    rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    unfold call_function. apply run_call_ordered; try assumption.
    - apply shape_of_call_related;
        [ symmetry; apply closure_of_ordered; exact Hstatics | | ].
      + intros called slots _. apply codes_agree_ordered. exact Hstatics.
      + intros tuple rest p _. symmetry.
        exact (eval_prim_same_static_values _ _ p [tuple] (proj2 Hstatics)).
    - intros partial. apply IHreturn. exact Hreturn. }
  induction c
    as [h vars conts recursive | exit | c IHc extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros c' vs Hc.
  - inversion Hc
      as [c0 | vm km km' h0 h' vars0 vars' conts0 conts' recursive0
                 recursive' Hnames Hh Hrec Hvars Hconts | | ];
      subst.
    + destruct (closure_ordered_itself h vars conts recursive)
        as (km' & Hnames & Hh & Hrec).
      exact (Hclosure vs [] [] km' h h vars vars conts conts recursive
        recursive Hnames Hh Hrec (envs_match_same _ vars)
        (conts_ordered_same _ conts)).
    + exact (Hclosure vs vm km km' h h' vars vars' conts conts' recursive
        recursive' Hnames Hh Hrec Hvars Hconts).
  - inversion Hc; subst. simpl. auto.
  - inversion Hc; subst; simpl; apply IHc;
      [ apply Same_ordered | assumption ].
  - inversion Hc
      as [c0 | |
         | remaining0 remaining_arity0 return_arity0 return_to0 return_to'
             exn_to0 exn_to' Hreturn Hexn];
      subst.
    + apply Hover; [ exact IHreturn | apply Same_ordered .. ].
    + apply Hover; assumption.
Qed.

(* The same call under static environments that are related, returning
   to continuations that are related. *)
Lemma call_function_ordered (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  statics_ordered statics statics' ->
  cont_ordered return_to return_to' ->
  cont_ordered exn_to exn_to' ->
  result_simulated state_ordered
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       statics' hp rd).
Proof.
  intros Hstatics Hreturn Hexn. unfold call_function.
  apply run_call_ordered; try assumption.
  - apply shape_of_call_related;
      [ symmetry; apply closure_of_ordered; exact Hstatics | | ].
    + intros called slots _. apply codes_agree_ordered. exact Hstatics.
    + intros tuple rest p _. symmetry.
      exact (eval_prim_same_static_values _ _ p [tuple] (proj2 Hstatics)).
  - intros partial. apply apply_cont_value_ordered; assumption.
Qed.

(* Where the first of two related states steps, the second steps to a
   related state, or the two leave by the same way with the same values,
   heap and record of reads. *)
Lemma ordered_step (s s' : state) :
  state_ordered s s' -> result_simulated state_ordered (step s) (step s').
Proof.
  intros [vm km e e' vars vars' conts conts' statics statics' hp rd He Hvars
            Hconts Hstatics].
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ],
    e'
    as [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
       | ap' | ac' | scrutinee' arms' | ];
    simpl in He; try discriminate; try rewrite !step_let_exec;
    autorewrite with step; simpl.
  - (* Let *)
    apply andb_true_iff in He as [Hd Hbody].
    apply andb_true_iff in Hd as [Hkind Hd].
    apply kind_eqb_eq in Hkind. subst kind'.
    assert (Hvars_d :
      envs_match (fun y => var_free_in_named y d) vm vars vars')
      by (fewer_variables Hvars; left; assumption).
    rewrite (exec_named_ordered vm vars vars' statics statics' hp rd d d'
      Hvars_d Hstatics Hd).
    destruct (exec_named statics' hp rd vars' d') as [[[v hp'] rd'] | ]; simpl;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); simpl; [ | exact I ].
    apply (State_ordered ((x, x') :: vm) km);
      [ exact Hbody | | exact Hconts | exact Hstatics ].
    refine (envs_match_cons _ _ _ _ _ _ _ _ _ Hvars). intros y Hy.
    destruct (Pos.eq_dec y x) as [Heq | Hne]; [ left; exact Heq | ].
    right. right. split; assumption.
  - (* Let_static *)
    apply andb_true_iff in He as [Hbindings Hbody].
    assert (Hvars_bindings :
      envs_match (fun y => Exists (var_free_in_static_binding y) bindings) vm
        vars vars')
      by (fewer_variables Hvars; left; assumption).
    pose proof (eval_static_bindings_ordered vm vars vars' bindings
      Hvars_bindings bindings' statics statics' Hbindings Hstatics)
      as Hextended.
    destruct (eval_static_bindings vars bindings statics),
      (eval_static_bindings vars' bindings' statics');
      try contradiction; simpl; [ | exact I ].
    apply (State_ordered vm km);
      [ exact Hbody | | exact Hconts | exact Hextended ].
    fewer_variables Hvars. right. assumption.
  - (* Let_cont_non_recursive *)
    apply andb_true_iff in He as [Hh Hbody].
    apply (State_ordered vm ((k, k') :: km));
      [ exact Hbody | fewer_variables Hvars; right; assumption |
      | exact Hstatics ].
    refine (conts_ordered_cons _ _ _ _ _ _ _ _ _ _ Hconts _).
    + intros j Hj.
      destruct (Pos.eq_dec j k) as [Heq | Hne]; [ left; exact Heq | ].
      right. right. split; assumption.
    + apply (Closure_ordered vm km km h h' vars vars' conts conts' [] []);
        [ reflexivity | exact Hh | reflexivity | | ].
      * fewer_variables Hvars.
        match goal with
        | Hy : _ ∨ _ |- _ => destruct Hy as [Hy | []]; left; exact Hy
        end.
      * intros j j' [[Hj | []] _]. apply Hconts. left. exact Hj.
  - (* Let_cont_recursive *)
    destruct (combine_names (map fst handlers) (map fst handlers') km)
      as [km' | ] eqn:Hnames; [ | discriminate ].
    apply andb_true_iff in He as [Hhandlers Hbody].
    rewrite <- (handlers_say_at_least_bind_statics vm km' handlers handlers'
      Hhandlers).
    destruct (handlers_bind_statics handlers); [ exact I | ].
    apply (State_ordered vm km');
      [ exact Hbody | fewer_variables Hvars; right; assumption |
      | exact Hstatics ].
    refine
      (rec_conts_ordered _ _ vm km km' vars vars' conts conts' handlers
         handlers' Hnames Hhandlers _ _ _ Hconts).
    + fewer_variables Hvars. left. assumption.
    + intros j Hj Hunbound. split; [ exact Hunbound | left; exact Hj ].
    + intros j Hj.
      destruct (lookup_none_or_in j handlers) as [Hunbound | Hbound];
        [ right | left; exact Hbound ].
      split; [ exact Hunbound | right; exact Hj ].
  - (* Apply *)
    unfold apply_eqb in He.
    apply andb_true_iff in He as [He Hreturn_arity].
    apply andb_true_iff in He as [He Hargs_arity].
    apply arity_eqb_eq in Hargs_arity, Hreturn_arity.
    apply andb_true_iff in He as [He _].
    apply andb_true_iff in He as [He Hextra].
    apply andb_true_iff in He as [He Hexn].
    apply andb_true_iff in He as [He Hreturn].
    apply andb_true_iff in He as [He Hargs].
    apply andb_true_iff in He as [Hcallee Hkind].
    apply call_kind_eqb_eq in Hkind.
    assert (Hvars_callee :
      envs_match (fun y => var_free_in_simple y (ap_callee ap)) vm vars vars')
      by (fewer_variables Hvars; left; assumption).
    assert (Hvars_args :
      envs_match (fun y => var_free_in_simples y (ap_args ap)) vm vars vars')
      by (fewer_variables Hvars; right; left; assumption).
    assert (Hvars_extra :
      envs_match (fun y => var_free_in_simples y (exn_extra_simples ap)) vm
        vars vars')
      by (fewer_variables Hvars; right; right; assumption).
    rewrite (eval_simple_match vm vars vars' _ _ Hvars_callee Hcallee),
      (eval_simples_match vm vars vars' _ Hvars_args _ Hargs),
      (eval_simples_match vm vars vars' _ Hvars_extra _ Hextra), <- Hkind,
      <- Hargs_arity, <- Hreturn_arity.
    destruct (eval_simple vars' (ap_callee ap')) as [callee | ];
      [ | exact I ].
    destruct (eval_simples vars' (ap_args ap')) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars' (exn_extra_simples ap')) as [extra | ];
      [ | exact I ].
    apply call_function_ordered;
      [ exact Hstatics
      | exact (Hconts _ _ (or_introl eq_refl) Hreturn)
      | apply with_extra_args_ordered;
        exact (Hconts _ _ (or_intror eq_refl) Hexn) ].
  - (* Apply_cont *)
    unfold apply_cont_eqb in He. apply andb_true_iff in He as [He Htrap].
    apply andb_true_iff in He as [Hk Hargs].
    rewrite (eval_simples_match vm vars vars' _ Hvars _ Hargs).
    destruct (eval_simples vars' (ac_args ac')) as [vs | ]; [ | exact I ].
    apply apply_cont_value_ordered;
      [ exact (Hconts _ _ (or_introl eq_refl) Hk) | exact Hstatics ].
  - (* Switch *)
    apply andb_true_iff in He as [Hscrutinee Harms].
    assert (Hvars_scrutinee :
      envs_match (fun y => var_free_in_simple y scrutinee) vm vars vars')
      by (fewer_variables Hvars; left; assumption).
    rewrite (eval_simple_match vm vars vars' _ _ Hvars_scrutinee Hscrutinee).
    destruct (eval_simple vars' scrutinee') as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    pose proof (find_arm_match vm km i arms arms' Harms) as Harm.
    destruct (find_arm i arms) as [ac | ] eqn:Hfind,
      (find_arm i arms') as [ac' | ];
      try contradiction; simpl; [ | exact I ].
    apply find_arm_in in Hfind.
    apply (State_ordered vm km); [ exact Harm | | | exact Hstatics ].
    + fewer_variables Hvars. right. apply Exists_exists.
      exists (i, ac). split; assumption.
    + fewer_continuations Hconts. apply Exists_exists.
      exists (i, ac). split; assumption.
  - (* Invalid *)
    exact I.
Qed.

(* So the second of two related states refines the first. *)
Theorem ordered_refines (s s' : state) :
  state_ordered s s' -> refines s s'.
Proof. apply (simulation_refines state_ordered ordered_step). Qed.

Theorem expr_says_at_least_sound : expr_says_at_least_sound_statement.
Proof.
  intros e e' Heq vars conts statics hp rd. apply ordered_refines.
  apply (State_ordered [] []);
    [ exact Heq | apply envs_match_same | apply conts_ordered_same
    | apply statics_ordered_same ].
Qed.
