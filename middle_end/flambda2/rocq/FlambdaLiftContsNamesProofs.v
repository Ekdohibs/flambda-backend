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

(* FlambdaLiftContsSpec.lift_handler_conts commutes with renaming: the
   lemma from which FlambdaLiftContsProofs.v proves
   [lift_handler_conts_equivariant_statement]. There is no statement
   here, and nothing here needs review.

   The pass compares names by equality only and makes none, so each of
   its functions gives, on renamed arguments, the renamed result or
   the same answer. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaNormaliseSpec FlambdaUnusedContsSpec FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaValidatorEquivarianceBaseProofs
  FlambdaUnaliasParamsSpec FlambdaGiveScrutineeSpec FlambdaLiftContsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists of names                                                           *)
(* ======================================================================== *)

Lemma trap_action_handler_renamed (r : renaming) (action : trap_action) :
  trap_action_handler (rename_trap_action r action)
  = rename_continuation r (trap_action_handler action).
Proof. destruct action; reflexivity. Qed.

Lemma distinct_variables_renamed (r : renaming) (names : list variable) :
  distinct (map (rename_variable r) names) = distinct names.
Proof.
  induction names as [ | y names IH]; [ reflexivity | ].
  cbn [map distinct]. rewrite among_rename_variable, IH. reflexivity.
Qed.

Lemma any_among_renamed (r : renaming) (params ps : list variable) :
  existsb (fun p => among p (map (rename_variable r) params))
    (map (rename_variable r) ps)
  = existsb (fun p => among p params) ps.
Proof.
  apply existsb_map_with. intros p. apply among_rename_variable.
Qed.

(* ======================================================================== *)
(* The applications given more arguments                                    *)
(* ======================================================================== *)

Lemma give_to_equivariant (r : renaming) (j : continuation)
    (ps : list variable) (ac : apply_cont) :
  give_to (rename_continuation r j) (map (rename_variable r) ps)
    (rename_apply_cont r ac)
  = rename_apply_cont r (give_to j ps ac).
Proof.
  unfold give_to, rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb (ac_continuation ac) j); [ | reflexivity ].
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite map_app, !map_map. reflexivity.
Qed.

Fixpoint give_equivariant (r : renaming) (j : continuation)
    (ps : list variable) (e : expr) {struct e} :
  give (rename_continuation r j) (map (rename_variable r) ps)
    (rename_expr r e)
  = rename_expr r (give j ps e)
with give_in_handler_equivariant (r : renaming) (j : continuation)
    (ps : list variable) (h : continuation_handler) {struct h} :
  give_in_handler (rename_continuation r j) (map (rename_variable r) ps)
    (rename_handler r h)
  = rename_handler r (give_in_handler j ps h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr give]; try reflexivity.
    + rewrite give_equivariant. reflexivity.
    + rewrite give_equivariant. reflexivity.
    + rewrite give_in_handler_equivariant, give_equivariant. reflexivity.
    + rewrite give_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map]. rewrite give_in_handler_equivariant, IH. reflexivity.
    + rewrite give_to_equivariant. reflexivity.
    + f_equal. induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map fst snd]. rewrite give_to_equivariant, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler give_in_handler].
    rewrite give_equivariant. reflexivity.
Qed.

Lemma applies_only_equivariant (r : renaming) (j : continuation) (ok : bool)
    (ac : apply_cont) :
  applies_only (rename_continuation r j) ok (rename_apply_cont r ac)
  = applies_only j ok ac.
Proof.
  unfold applies_only, rename_apply_cont.
  cbn [ac_trap_action ac_continuation].
  rewrite rename_continuation_eqb. f_equal.
  destruct (ac_trap_action ac) as [action | ]; [ | reflexivity ].
  cbn [option_map].
  rewrite trap_action_handler_renamed, rename_continuation_eqb.
  reflexivity.
Qed.

Fixpoint only_applied_equivariant (r : renaming) (j : continuation)
    (ps : list variable) (ok : bool) (e : expr) {struct e} :
  only_applied (rename_continuation r j) (map (rename_variable r) ps) ok
    (rename_expr r e)
  = only_applied j ps ok e
with handler_only_applied_equivariant (r : renaming) (j : continuation)
    (ps : list variable) (ok : bool) (h : continuation_handler) {struct h} :
  handler_only_applied (rename_continuation r j) (map (rename_variable r) ps)
    ok (rename_handler r h)
  = handler_only_applied j ps ok h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr only_applied]; try reflexivity.
    + rewrite among_rename_variable. apply only_applied_equivariant.
    + apply only_applied_equivariant.
    + rewrite rename_continuation_eqb, handler_only_applied_equivariant,
        only_applied_equivariant.
      reflexivity.
    + rewrite only_applied_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite rename_continuation_eqb, handler_only_applied_equivariant, IH.
      reflexivity.
    + cbn [rename_apply ap_continuation ap_exn_continuation].
      destruct ps as [ | p0 ps0]; cbn [map];
        rewrite !rename_continuation_eqb; reflexivity.
    + apply applies_only_equivariant.
    + induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map forallb fst snd].
      rewrite applies_only_equivariant, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_only_applied].
    rewrite any_among_renamed. apply only_applied_equivariant.
Qed.

(* ======================================================================== *)
(* The continuations of a handler, moved out                                *)
(* ======================================================================== *)

Definition rename_read (r : renaming)
    (read : list (variable * flambda_kind)) : list (variable * flambda_kind) :=
  map (fun '(p, kind) => (rename_variable r p, kind)) read.

Lemma map_fst_rename_read (r : renaming)
    (read : list (variable * flambda_kind)) :
  map fst (rename_read r read) = map (rename_variable r) (map fst read).
Proof.
  unfold rename_read. rewrite !map_map. apply map_ext. intros [p kind].
  reflexivity.
Qed.

Lemma map_snd_rename_read (r : renaming)
    (read : list (variable * flambda_kind)) :
  map snd (rename_read r read) = map snd read.
Proof.
  unfold rename_read. rewrite map_map. apply map_ext. intros [p kind].
  reflexivity.
Qed.

Lemma read_params_equivariant (r : renaming) (e : expr)
    (params : list variable) :
  ∀ kinds : list flambda_kind,
    read_params (map (rename_variable r) params) kinds (rename_expr r e)
    = rename_read r (read_params params kinds e).
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds];
    cbn [map read_params]; try reflexivity.
  rewrite var_free_b_equivariant_aux, IH.
  destruct (var_free_b p e); reflexivity.
Qed.

Definition rename_moved (r : renaming)
    (moved : list (continuation * list variable))
    : list (continuation * list variable) :=
  map (fun '(j, ps) => (rename_continuation r j, map (rename_variable r) ps))
    moved.

Lemma give_all_equivariant (r : renaming)
    (moved : list (continuation * list variable)) (e : expr) :
  give_all (rename_moved r moved) (rename_expr r e)
  = rename_expr r (give_all moved e).
Proof.
  unfold give_all, rename_moved.
  induction moved as [ | [j ps] moved IH]; [ reflexivity | ].
  cbn [map fold_right]. rewrite IH. apply give_equivariant.
Qed.

Definition rename_lifted (r : renaming)
    (lifted : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(j, h) => (rename_continuation r j, rename_handler r h)) lifted.

Definition rename_spine (r : renaming)
    (result : option (list (continuation * continuation_handler) * expr))
    : option (list (continuation * continuation_handler) * expr) :=
  option_map
    (fun '(lifted, switch) => (rename_lifted r lifted, rename_expr r switch))
    result.

Lemma lift_spine_equivariant (r : renaming) (body : expr)
    (params : list variable) (kinds : list flambda_kind) (handler : expr) :
  ∀ moved : list (continuation * list variable),
    lift_spine (rename_expr r body) (map (rename_variable r) params) kinds
      (rename_moved r moved) (rename_expr r handler)
    = rename_spine r (lift_spine body params kinds moved handler).
Proof.
  induction handler
    as [ y kind d rest _ | bindings rest _ | j [q mdj h] rest IH
       | handlers rest _ | ap | ac | scrutinee arms | ];
    intros moved; cbn [rename_expr rename_handler lift_spine];
    try reflexivity.
  - (* a continuation of the handler *)
    rewrite !give_all_equivariant, !read_params_equivariant,
      !map_fst_rename_read, map_snd_rename_read, only_applied_equivariant,
      !cont_free_b_equivariant_aux, <- map_app, distinct_variables_renamed.
    set (h' := give_all moved h).
    set (read := read_params params kinds h').
    destruct
      (negb (hm_is_exn_handler mdj) && distinct (params ++ q)
       && negb (cont_free_b j body) && negb (cont_free_b j h')
       && only_applied j (map fst read) true (give_all moved rest));
      [ | reflexivity ].
    pose proof (IH ((j, map fst read) :: moved)) as Hrest.
    cbn [rename_moved map] in Hrest. fold (rename_moved r moved) in Hrest.
    rewrite Hrest.
    destruct (lift_spine body params kinds ((j, map fst read) :: moved) rest)
      as [[others switch] | ];
      [ | reflexivity ].
    cbn [rename_spine option_map rename_lifted map rename_handler].
    rewrite map_app. reflexivity.
  - (* the [Switch] *)
    destruct scrutinee as [p | sym | c]; cbn [rename_simple];
      try reflexivity.
    rewrite among_rename_variable.
    destruct (among p params); [ | reflexivity ].
    cbn [rename_spine option_map rename_lifted map].
    exact
      (f_equal (fun e => Some ([], e))
         (give_all_equivariant r moved (Switch (Var p) arms))).
Qed.

Lemma position_equivariant (r : renaming) (p : variable)
    (params : list variable) :
  position (rename_variable r p) (map (rename_variable r) params)
  = position p params.
Proof.
  induction params as [ | p' params IH]; [ reflexivity | ].
  cbn [map position]. rewrite rename_variable_eqb, IH. reflexivity.
Qed.

Lemma tested_equivariant (r : renaming) (params : list variable)
    (handler : expr) :
  tested (map (rename_variable r) params) (rename_expr r handler)
  = tested params handler.
Proof.
  induction handler
    as [ y kind d rest _ | bindings rest _ | j hj rest IH | handlers rest _
       | ap | ac | scrutinee arms | ];
    cbn [rename_expr tested]; try reflexivity; [ exact IH | ].
  destruct scrutinee as [p | sym | c]; cbn [rename_simple]; try reflexivity.
  apply position_equivariant.
Qed.

Lemma gives_immediate_equivariant (r : renaming) (k : continuation) (i : nat)
    (ac : apply_cont) :
  gives_immediate (rename_continuation r k) i (rename_apply_cont r ac)
  = gives_immediate k i ac.
Proof.
  unfold gives_immediate, rename_apply_cont.
  cbn [ac_trap_action ac_continuation ac_args].
  rewrite rename_continuation_eqb, nth_error_map.
  f_equal.
  - destruct (ac_trap_action ac) as [action | ]; [ | reflexivity ].
    cbn [option_map].
    rewrite trap_action_handler_renamed, rename_continuation_eqb.
    reflexivity.
  - destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
    destruct (nth_error (ac_args ac) i) as [[y | sym | c] | ]; reflexivity.
Qed.

Fixpoint all_give_immediate_equivariant (r : renaming) (k : continuation)
    (i : nat) (e : expr) {struct e} :
  all_give_immediate (rename_continuation r k) i (rename_expr r e)
  = all_give_immediate k i e
with handler_all_give_immediate_equivariant (r : renaming) (k : continuation)
    (i : nat) (h : continuation_handler) {struct h} :
  handler_all_give_immediate (rename_continuation r k) i (rename_handler r h)
  = handler_all_give_immediate k i h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr all_give_immediate]; try reflexivity.
    + apply all_give_immediate_equivariant.
    + apply all_give_immediate_equivariant.
    + rewrite rename_continuation_eqb, handler_all_give_immediate_equivariant,
        all_give_immediate_equivariant.
      reflexivity.
    + rewrite all_give_immediate_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite rename_continuation_eqb, handler_all_give_immediate_equivariant,
        IH.
      reflexivity.
    + cbn [rename_apply ap_continuation ap_exn_continuation].
      rewrite !rename_continuation_eqb. reflexivity.
    + apply gives_immediate_equivariant.
    + induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map forallb fst snd].
      rewrite gives_immediate_equivariant, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_all_give_immediate].
    apply all_give_immediate_equivariant.
Qed.

(* ======================================================================== *)
(* Whether the tested parameter is a scrutinee under another name           *)
(* ======================================================================== *)

Lemma filter_under_map {A B : Type} (keep' : B -> bool) (g : A -> B)
    (keep : A -> bool) (l : list A) :
  (∀ a : A, keep' (g a) = keep a) ->
  filter keep' (map g l) = map g (filter keep l).
Proof.
  intros Hkeep. induction l as [ | a l IH]; cbn [map filter];
    [ reflexivity | ].
  rewrite Hkeep, IH. destruct (keep a); reflexivity.
Qed.

Definition rename_known (r : renaming) (known : list (variable * Z))
    : list (variable * Z) :=
  map (fun '(x, c) => (rename_variable r x, c)) known.

Lemma first_arm_of_renamed (r : renaming) (k : continuation)
    (arms : list (Z * apply_cont)) :
  first_arm_of (rename_continuation r k)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = first_arm_of k arms.
Proof.
  unfold first_arm_of.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map first_answer fst snd rename_apply_cont ac_continuation].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb (ac_continuation ac) k); [ reflexivity | exact IH ].
Qed.

Lemma only_arm_renamed (r : renaming) (k : continuation) (c : Z)
    (arms : list (Z * apply_cont)) :
  only_arm (rename_continuation r k) c
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = only_arm k c arms.
Proof.
  unfold only_arm.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map forallb fst snd rename_apply_cont ac_continuation].
  rewrite rename_continuation_eqb, IH. reflexivity.
Qed.

Fixpoint tested_through_equivariant (r : renaming) (e : expr) {struct e} :
  tested_through (rename_expr r e)
  = option_map (rename_variable r) (tested_through e).
Proof.
  destruct e as [y kind d body | bindings body | k' hh body | handlers body
                | ap | ac | scrutinee arms | ];
    cbn [rename_expr tested_through]; try reflexivity;
    try apply tested_through_equivariant.
  destruct scrutinee as [x | sym | c]; reflexivity.
Qed.

Fixpoint arm_through_equivariant (r : renaming) (k' : continuation)
    (body : expr) {struct body} :
  arm_through (rename_continuation r k') (rename_expr r body)
  = arm_through k' body.
Proof.
  destruct body as [y kind d body | bindings body | k'' hh body
                   | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr arm_through]; try reflexivity;
    try apply arm_through_equivariant.
  - rewrite rename_continuation_eqb, cont_free_in_handler_b_equivariant,
      arm_through_equivariant.
    reflexivity.
  - rewrite first_arm_of_renamed.
    destruct (first_arm_of k' arms) as [c | ]; [ | reflexivity ].
    rewrite only_arm_renamed. reflexivity.
Qed.

Lemma facts_in_equivariant (r : renaming) (k' : continuation) (body : expr)
    (known : list (variable * Z)) :
  facts_in (rename_continuation r k') (rename_expr r body)
    (rename_known r known)
  = rename_known r (facts_in k' body known).
Proof.
  unfold facts_in.
  rewrite tested_through_equivariant, arm_through_equivariant.
  destruct (tested_through body) as [x | ]; [ | reflexivity ].
  cbn [option_map].
  destruct (arm_through k' body) as [c | ]; reflexivity.
Qed.

Lemma forgetting_equivariant (r : renaming) (ys : list variable)
    (known : list (variable * Z)) :
  forgetting (map (rename_variable r) ys) (rename_known r known)
  = rename_known r (forgetting ys known).
Proof.
  unfold forgetting, rename_known.
  apply filter_under_map. intros [y c].
  rewrite among_rename_variable. reflexivity.
Qed.

Lemma in_arm_equivariant (r : renaming) (scrutinee : simple) (d : Z)
    (known : list (variable * Z)) :
  in_arm (rename_simple r scrutinee) d (rename_known r known)
  = rename_known r (in_arm scrutinee d known).
Proof. destruct scrutinee; reflexivity. Qed.

Lemma keep_known_equivariant (r : renaming) (k : continuation) (i : nat)
    (known : list (variable * Z)) (ac : apply_cont) (cands : list variable) :
  keep_known (rename_continuation r k) i (rename_known r known)
    (rename_apply_cont r ac) (map (rename_variable r) cands)
  = map (rename_variable r) (keep_known k i known ac cands).
Proof.
  unfold keep_known.
  replace (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)) by reflexivity.
  replace (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)) by reflexivity.
  rewrite rename_continuation_eqb, nth_error_map.
  destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
  destruct (nth_error (ac_args ac) i) as [[y | sym | cst] | ];
    try reflexivity.
  destruct cst; try reflexivity.
  cbn [option_map rename_simple].
  apply filter_under_map. intros x.
  unfold rename_known. apply existsb_map_with. intros [y d].
  rewrite rename_variable_eqb. reflexivity.
Qed.

Fixpoint still_known_equivariant (r : renaming) (k : continuation) (i : nat)
    (e : expr) {struct e} :
  ∀ (known : list (variable * Z)) (cands : list variable),
    still_known (rename_continuation r k) i (rename_known r known)
      (rename_expr r e) (map (rename_variable r) cands)
    = map (rename_variable r) (still_known k i known e cands)
with handler_still_known_equivariant (r : renaming) (k : continuation)
    (i : nat) (h : continuation_handler) {struct h} :
  ∀ (known : list (variable * Z)) (cands : list variable),
    handler_still_known (rename_continuation r k) i (rename_known r known)
      (rename_handler r h) (map (rename_variable r) cands)
    = map (rename_variable r) (handler_still_known k i known h cands).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros known cands; cbn [rename_expr still_known]; try reflexivity.
    + pose proof (forgetting_equivariant r [y] known) as Hforget.
      cbn [map] in Hforget. rewrite Hforget.
      apply still_known_equivariant.
    + apply still_known_equivariant.
    + rewrite facts_in_equivariant, handler_still_known_equivariant.
      apply still_known_equivariant.
    + rewrite <- still_known_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map fold_right fst snd]. rewrite IH.
      apply handler_still_known_equivariant.
    + apply keep_known_equivariant.
    + induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
      cbn [map fold_right fst snd].
      rewrite IH, in_arm_equivariant. apply keep_known_equivariant.
  - destruct h as [params metadata handler]; intros known cands.
    cbn [rename_handler handler_still_known].
    rewrite forgetting_equivariant. apply still_known_equivariant.
Qed.

Fixpoint scrutinees_equivariant (r : renaming) (e : expr) {struct e} :
  scrutinees (rename_expr r e) = map (rename_variable r) (scrutinees e)
with handler_scrutinees_equivariant (r : renaming) (h : continuation_handler)
    {struct h} :
  handler_scrutinees (rename_handler r h)
  = map (rename_variable r) (handler_scrutinees h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr scrutinees]; try reflexivity.
    + apply scrutinees_equivariant.
    + apply scrutinees_equivariant.
    + rewrite handler_scrutinees_equivariant, scrutinees_equivariant, map_app.
      reflexivity.
    + rewrite scrutinees_equivariant, map_app. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map flat_map fst snd].
      rewrite handler_scrutinees_equivariant, IH, map_app. reflexivity.
    + destruct scrutinee as [x | sym | c]; reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_scrutinees].
    apply scrutinees_equivariant.
Qed.

Lemma number_of_a_scrutinee_equivariant (r : renaming) (k : continuation)
    (i : nat) (body : expr) :
  number_of_a_scrutinee (rename_continuation r k) i (rename_expr r body)
  = number_of_a_scrutinee k i body.
Proof.
  unfold number_of_a_scrutinee.
  rewrite scrutinees_equivariant.
  pose proof (still_known_equivariant r k i body [] (scrutinees body))
    as Hknown.
  cbn [rename_known map] in Hknown. rewrite Hknown.
  destruct (still_known k i [] body (scrutinees body)); reflexivity.
Qed.

(* ======================================================================== *)
(* Whether an application gives variables only where its arm reads them     *)
(* ======================================================================== *)

Lemma only_read_given_equivariant (r : renaming) (target : apply_cont)
    (params : list variable) :
  ∀ args : list simple,
    only_read_given (map (rename_variable r) params)
      (map (rename_simple r) args) (rename_apply_cont r target)
    = only_read_given params args target.
Proof.
  induction params as [ | p params IH]; intros [ | a args];
    cbn [map only_read_given]; try reflexivity.
  rewrite IH. f_equal.
  destruct a as [x | sym | c]; cbn [rename_simple]; try reflexivity.
  replace (ac_args (rename_apply_cont r target))
    with (map (rename_simple r) (ac_args target)) by reflexivity.
  apply existsb_map_with. intros [y | sym | c]; cbn [rename_simple];
    try reflexivity.
  apply rename_variable_eqb.
Qed.

Lemma gives_what_is_read_equivariant (r : renaming) (k : continuation)
    (i : nat) (params : list variable) (arms : list (Z * apply_cont))
    (ac : apply_cont) :
  gives_what_is_read (rename_continuation r k) i
    (map (rename_variable r) params)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    (rename_apply_cont r ac)
  = gives_what_is_read k i params arms ac.
Proof.
  unfold gives_what_is_read.
  replace (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)) by reflexivity.
  replace (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)) by reflexivity.
  rewrite rename_continuation_eqb, nth_error_map.
  destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
  destruct (nth_error (ac_args ac) i) as [[y | sym | cst] | ];
    try reflexivity.
  destruct cst; try reflexivity.
  cbn [option_map rename_simple].
  rewrite find_arm_renamed.
  destruct (find_arm _ arms) as [target | ]; [ | reflexivity ].
  cbn [option_map]. apply only_read_given_equivariant.
Qed.

Fixpoint all_give_read_equivariant (r : renaming) (k : continuation)
    (i : nat) (params : list variable) (arms : list (Z * apply_cont))
    (e : expr) {struct e} :
  all_give_read (rename_continuation r k) i (map (rename_variable r) params)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    (rename_expr r e)
  = all_give_read k i params arms e
with handler_all_give_read_equivariant (r : renaming) (k : continuation)
    (i : nat) (params : list variable) (arms : list (Z * apply_cont))
    (h : continuation_handler) {struct h} :
  handler_all_give_read (rename_continuation r k) i
    (map (rename_variable r) params)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    (rename_handler r h)
  = handler_all_give_read k i params arms h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms' | ];
      cbn [rename_expr all_give_read]; try reflexivity.
    + apply all_give_read_equivariant.
    + apply all_give_read_equivariant.
    + rewrite handler_all_give_read_equivariant, all_give_read_equivariant.
      reflexivity.
    + rewrite all_give_read_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb fst snd].
      rewrite handler_all_give_read_equivariant, IH. reflexivity.
    + apply gives_what_is_read_equivariant.
    + induction arms' as [ | [n ac] arms' IH]; [ reflexivity | ].
      cbn [map forallb fst snd].
      rewrite gives_what_is_read_equivariant, IH. reflexivity.
  - destruct h as [ps metadata handler].
    cbn [rename_handler handler_all_give_read].
    apply all_give_read_equivariant.
Qed.

Lemma read_where_given_equivariant (r : renaming) (k : continuation)
    (i : nat) (params : list variable) (switch body : expr) :
  read_where_given (rename_continuation r k) i
    (map (rename_variable r) params) (rename_expr r switch)
    (rename_expr r body)
  = read_where_given k i params switch body.
Proof.
  destruct switch; cbn [rename_expr read_where_given]; try reflexivity.
  apply all_give_read_equivariant.
Qed.

(* ======================================================================== *)
(* The [Let_cont] with the continuations of its handler moved out           *)
(* ======================================================================== *)

Lemma bind_lifted_equivariant (r : renaming)
    (lifted : list (continuation * continuation_handler)) (e : expr) :
  bind_lifted (rename_lifted r lifted) (rename_expr r e)
  = rename_expr r (bind_lifted lifted e).
Proof.
  unfold bind_lifted, rename_lifted.
  induction lifted as [ | [j hj] lifted IH]; [ reflexivity | ].
  cbn [map fold_right rename_expr]. rewrite IH. reflexivity.
Qed.

Lemma lift_conts_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) :
  lift_conts (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = option_map (rename_expr r) (lift_conts k h body).
Proof.
  destruct h as [params metadata handler].
  cbn [rename_handler lift_conts].
  rewrite tested_equivariant.
  destruct (tested params handler) as [i | ]; [ | reflexivity ].
  rewrite distinct_variables_renamed, length_map,
    all_give_immediate_equivariant, number_of_a_scrutinee_equivariant.
  destruct
    (negb (hm_is_exn_handler metadata) && distinct params
     && Nat.eqb (length (hm_params_kinds metadata)) (length params)
     && all_give_immediate k i body
     && negb (number_of_a_scrutinee k i body));
    [ | reflexivity ].
  pose proof
    (lift_spine_equivariant r body params (hm_params_kinds metadata) handler
       [])
    as Hspine.
  cbn [rename_moved map] in Hspine. rewrite Hspine.
  destruct (lift_spine body params (hm_params_kinds metadata) [] handler)
    as [[[ | [j hj] others] switch] | ];
    try reflexivity.
  cbn [rename_spine option_map rename_lifted map].
  fold (rename_lifted r others).
  rewrite read_where_given_equivariant.
  destruct (read_where_given k i params switch body); [ | reflexivity ].
  cbn [option_map].
  pose proof
    (bind_lifted_equivariant r ((j, hj) :: others)
       (Let_cont_non_recursive k (Handler params metadata switch) body))
    as Hbind.
  cbn [rename_lifted map rename_expr rename_handler] in Hbind.
  fold (rename_lifted r others) in Hbind.
  rewrite Hbind. reflexivity.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Fixpoint lift_handler_conts_equivariant_aux (r : renaming) (e : expr) :
  lift_handler_conts (rename_expr r e) = rename_expr r (lift_handler_conts e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler lift_handler_conts]; try reflexivity.
  - rewrite lift_handler_conts_equivariant_aux. reflexivity.
  - rewrite lift_handler_conts_equivariant_aux. reflexivity.
  - rewrite !lift_handler_conts_equivariant_aux.
    pose proof
      (lift_conts_equivariant r k
         (Handler params metadata (lift_handler_conts handler))
         (lift_handler_conts body))
      as Hlift.
    cbn [rename_handler] in Hlift. rewrite Hlift.
    destruct
      (lift_conts k (Handler params metadata (lift_handler_conts handler))
         (lift_handler_conts body));
      reflexivity.
  - rewrite lift_handler_conts_equivariant_aux. reflexivity.
Qed.

Lemma lift_handler_conts_renamed : lift_handler_conts_equivariant_statement.
Proof. exact lift_handler_conts_equivariant_aux. Qed.
