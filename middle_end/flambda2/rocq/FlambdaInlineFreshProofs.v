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

(* Proofs of the statements of FlambdaInlineFreshSpec.v. Nothing here
   needs review.

   The refinement is three steps. The binders of closed code may be
   renamed where the code is bound (FlambdaNominalProofs.
   rename_code_binders), so the call of the code is equivalent to the
   call of the code opened at the fresh variables. The opened code
   meets FlambdaApplySpec.inlinable, its variables being the fresh
   ones ([opened_inlinable]), so its call refines its body inlined
   (FlambdaApplyProofs.inline_let_code). And the code is renamed back
   where it is bound. Of a state it is the same three steps, with the
   code replaced by the opened code in the static environment
   ([replace_code]): the two static environments have, at each code
   id, code that the comparison accepts, so a state under the one is
   equivalent to the state under the other
   (FlambdaCompareProofs.matched_equivalent).

   The choice. The result is a run of [Let]s that bind the fresh
   variables, around the body of the code renamed. For two lists of
   fresh variables that have no variable in common, the swap of the
   one with the other fixes the free variables of the first result,
   which are those of the call, and sends it to the second result up to
   the body, where the two renamings of the body of the code agree on
   its free variables and so give alpha-equivalent expressions
   ([alpha_renamings_agree]). Two lists that have variables in common
   are both compared with a third. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaContextSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaBindersSpec FlambdaBindersProofs
  FlambdaFreeNamesSpec FlambdaFreeNamesProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaNominalInstancesSpec FlambdaNominalInstancesProofs
  FlambdaCompareSpec FlambdaCompareProofs FlambdaCompareAlphaSpec
  FlambdaCompareAlphaProofs FlambdaApplySpec FlambdaApplyProofs
  FlambdaFreshQuantifierSpec
  FlambdaFreshQuantifierProofs FlambdaFreshBindersSpec
  FlambdaFreshBindersProofs FlambdaInlineFreshSpec.
Import ListNotations.

(* ======================================================================== *)
(* Code opened at fresh variables                                           *)
(* ======================================================================== *)

Lemma open_code_body (fresh : list variable) (c : code) :
  code_body (open_code fresh c)
  = rename_expr (swap_variables (code_variables c) fresh) (code_body c).
Proof. destruct c; reflexivity. Qed.

Lemma open_code_return (fresh : list variable) (c : code) :
  code_return_continuation (open_code fresh c) = code_return_continuation c.
Proof. destruct c; reflexivity. Qed.

Lemma open_code_exn (fresh : list variable) (c : code) :
  code_exn_continuation (open_code fresh c) = code_exn_continuation c.
Proof. destruct c; reflexivity. Qed.

Lemma open_code_arity (fresh : list variable) (c : code) :
  length (code_params (open_code fresh c)) = length (code_params c).
Proof. destruct c. apply length_map. Qed.

Lemma open_code_kinds (fresh : list variable) (c : code) :
  code_params_kinds (open_code fresh c) = code_params_kinds c.
Proof. destruct c; reflexivity. Qed.

(* The variables of the opened code are the fresh ones. *)
Lemma open_code_variables (fresh : list variable) (c : code) :
  length fresh = length (code_variables c) ->
  distinct (code_variables c) = true ->
  distinct_outside (code_variables c) fresh ->
  code_variables (open_code fresh c) = fresh.
Proof.
  intros Hlength Hdistinct Houtside.
  pose proof
    (rename_variables_swap (code_variables c) fresh Hlength Hdistinct
       Houtside)
    as Hswap.
  destruct c as [params my_closure ret exn metadata body]. exact Hswap.
Qed.

(* The free variables of closed code are among its variables. *)
Lemma closed_code_variables (c : code) (x : variable) :
  code_is_closed c = true ->
  among x (free_variables_of_expr (code_body c)) = true ->
  among x (code_variables c) = true.
Proof.
  intros Hclosed Hx.
  destruct (proj1 (closed_code c) Hclosed) as [Hamong _].
  unfold names_among in Hamong.
  apply andb_true_iff in Hamong as [Hamong _].
  exact (proj1 (all_among _ _) Hamong x Hx).
Qed.

Lemma opened_inlinable (ap : apply_expr) (c : code) (r : renaming)
    (fresh : list variable) :
  inlinable_at_fresh ap c r ->
  length fresh = length (code_variables c) ->
  distinct_outside
    (code_variables c ++ free_variables_of_simples (ap_args ap)) fresh ->
  inlinable ap (open_code fresh c) r.
Proof.
  intros
    (Hclosed & Hdistinct & Hvars & Hret & Hexn & Hextra & Harity & Hfull
     & Hstatics & Htupled)
    Hlength Houtside.
  apply distinct_outside_app in Houtside as [Houtside [_ Hargs]].
  pose proof (open_code_variables fresh c Hlength Hdistinct Houtside)
    as Hvariables.
  assert (Hclosed' : code_is_closed (open_code fresh c) = true)
    by exact (eq_trans (code_is_closed_renamed _ c) Hclosed).
  destruct (proj1 (closed_code _) Hclosed') as [Hamong Hclosed_body].
  unfold names_among in Hamong.
  apply andb_true_iff in Hamong as [_ Hamong_continuations].
  (* A variable of the opened code is fresh, so it is not free in the
     arguments. *)
  assert (Hnot_free :
    ∀ x : variable,
      among x (code_variables (open_code fresh c)) = true ->
      ~ var_free_in_simples x (ap_args ap)).
  { intros x Hx Hfree. apply var_free_in_simples_among in Hfree.
    assert (Hfresh : among x fresh = true)
      by exact (eq_trans (f_equal (among x) (eq_sym Hvariables)) Hx).
    pose proof (Hargs x Hfresh) as Hnot. congruence. }
  split.
  { intros x Hx. apply (var_free_agrees x _ Hclosed_body) in Hx.
    apply (closed_code_variables _ x Hclosed') in Hx.
    unfold code_variables in Hx. rewrite among_cons in Hx.
    apply orb_true_iff in Hx as [Hx | Hx];
      [ left; apply Pos.eqb_eq, Hx | right; apply among_In, Hx ]. }
  split.
  { intros k Hk. apply (cont_free_agrees k _ Hclosed_body) in Hk.
    apply (proj1 (all_among _ _) Hamong_continuations) in Hk.
    rewrite !among_cons in Hk.
    apply orb_true_iff in Hk as [Hk | Hk];
      [ left; apply Pos.eqb_eq, Hk | ].
    apply orb_true_iff in Hk as [Hk | Hk];
      [ right; apply Pos.eqb_eq, Hk | discriminate Hk ]. }
  split; [ exact Hvars | ].
  split; [ rewrite open_code_return; exact Hret | ].
  split; [ rewrite open_code_exn; exact Hexn | ].
  split; [ exact Hextra | ].
  split.
  { apply Hnot_free. unfold code_variables.
    rewrite among_cons, Pos.eqb_refl. reflexivity. }
  split.
  { intros param Hin. apply Hnot_free. unfold code_variables.
    apply among_In in Hin. rewrite among_cons, Hin. apply orb_true_r. }
  split; [ rewrite open_code_arity; exact Harity | ].
  split; [ destruct c; exact Hfull | ].
  split; [ | destruct c; exact Htupled ].
  rewrite open_code_body, rename_binds_statics. exact Hstatics.
Qed.

(* ======================================================================== *)
(* A call refines its code inlined at fresh variables                       *)
(* ======================================================================== *)

Theorem inline_at : inline_at_statement.
Proof.
  intros ap called c r fresh Hkind Hinlinable Hlength Houtside.
  pose proof Hinlinable as (Hclosed & _).
  pose proof (opened_inlinable ap c r fresh Hinlinable Hlength Houtside)
    as Hopened.
  pose proof
    (rename_code_binders (swap_variables (code_variables c) fresh) [] called
       c [])
    as Hbinders.
  intros vars conts statics hp rd.
  apply
    (refines_trans _
       (State (Let_static [Bind_code called (open_code fresh c)] (Apply ap))
          vars conts statics hp rd)).
  - apply equivalent_refines.
    exact (Hbinders (Apply ap) Hclosed vars conts statics hp rd_).
  - apply
      (refines_trans _
         (State
            (Let_static [Bind_code called (open_code fresh c)]
               (inlined_at fresh c r ap))
            vars conts statics hp rd)).
    + exact
        (inline_let_code ap called (open_code fresh c) r Hkind Hopened vars
           conts statics hp rd_).
    + apply equivalent_refines, equivalent_sym.
      exact
        (Hbinders (inlined_at fresh c r ap) Hclosed vars conts statics hp rd_).
Qed.

(* ======================================================================== *)
(* The same of a state                                                      *)
(* ======================================================================== *)

(* The first binding of a code id replaced by other code. *)
Fixpoint replace_code (called : code_id) (replacement : code)
    (codes : list (code_id * code)) : list (code_id * code) :=
  match codes with
  | [] => []
  | (id, c) :: codes =>
    if Pos.eqb called id
    then (id, replacement) :: codes
    else (id, c) :: replace_code called replacement codes
  end.

Lemma lookup_replace_code (called : code_id) (c replacement : code)
    (codes : list (code_id * code)) :
  lookup called codes = Some c ->
  lookup called (replace_code called replacement codes) = Some replacement.
Proof.
  induction codes as [ | [id first] codes IH]; cbn [lookup replace_code];
    [ discriminate | ].
  destruct (Pos.eqb called id) eqn:Hid; cbn [lookup]; rewrite Hid;
    [ reflexivity | exact IH ].
Qed.

Definition codes_match (codes codes' : list (code_id * code)) : Prop :=
  Forall2
    (fun ic ic' => fst ic = fst ic' ∧ code_eqb (snd ic) (snd ic') = true)
    codes codes'.

Lemma codes_match_same (codes : list (code_id * code)) :
  codes_match codes codes.
Proof.
  induction codes as [ | ic codes IH]; constructor; [ | exact IH ].
  split; [ reflexivity | apply code_eqb_refl ].
Qed.

(* The code replaced, and what replaces it, are accepted by the
   comparison in the order given, or in the other order. *)
Lemma replace_code_match (called : code_id) (c replacement : code)
    (codes : list (code_id * code)) :
  lookup called codes = Some c ->
  code_eqb c replacement = true ->
  codes_match codes (replace_code called replacement codes)
  ∧ codes_match (replace_code called replacement codes) codes.
Proof.
  intros Hcode Heq.
  induction codes as [ | [id first] codes IH]; cbn [lookup replace_code] in *;
    [ discriminate Hcode | ].
  destruct (Pos.eqb called id).
  - injection Hcode as ->.
    split; (constructor; [ | apply codes_match_same ]);
      (split; [ reflexivity | ]); cbn [snd];
      [ exact Heq | exact (code_eqb_flip _ _ Heq) ].
  - destruct (IH Hcode) as [Hforwards Hbackwards].
    split; (constructor; [ | assumption ]);
      (split; [ reflexivity | apply code_eqb_refl ]).
Qed.

(* A state under two static environments that match. *)
Lemma statics_matched_equivalent (e : expr) (vars : env) (conts : cont_env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  statics_match statics statics' ->
  equivalent (State e vars conts statics hp rd)
    (State e vars conts statics' hp rd).
Proof.
  intros Hmatch. apply matched_equivalent. apply (State_matches [] []).
  - apply expr_eqb_refl; apply reflexive_map_nil.
  - apply envs_match_same.
  - apply conts_match_same.
  - exact Hmatch.
Qed.

Theorem inline_at_state : inline_at_state_statement.
Proof.
  intros ap called c r fresh vars conts statics hp rd Hkind Hcode Hinlinable
    Hlength
    Houtside.
  pose proof Hinlinable as (Hclosed & _).
  pose proof (opened_inlinable ap c r fresh Hinlinable Hlength Houtside)
    as Hopened.
  set (statics' :=
    Static_env
      (replace_code called (open_code fresh c) (static_code statics))
      (static_values statics)).
  destruct
    (replace_code_match called c (open_code fresh c) (static_code statics)
       Hcode (rename_code_eqb _ c Hclosed))
    as [Hforwards Hbackwards].
  apply (refines_trans _ (State (Apply ap) vars conts statics' hp rd)).
  - apply equivalent_refines, statics_matched_equivalent.
    split; [ exact Hforwards | reflexivity ].
  - apply
      (refines_trans _
         (State (inlined_at fresh c r ap) vars conts statics' hp rd)).
    + apply (inline ap called (open_code fresh c) r);
        [ exact Hkind | | exact Hopened ].
      exact (lookup_replace_code called c _ _ Hcode).
    + apply equivalent_refines, statics_matched_equivalent.
      split; [ exact Hbackwards | reflexivity ].
Qed.

(* The variables that [fresh_names] makes will do. *)
Lemma fresh_for_code (avoid free : list variable) (c : code) :
  FlambdaMapSpec.all_among free avoid = true ->
  distinct_outside (code_variables c ++ free)
    (fresh_names (length (code_variables c)) (avoid ++ code_variables c)).
Proof.
  intros Hfree.
  destruct
    (fresh_names_outside (length (code_variables c))
       (avoid ++ code_variables c))
    as [Hdistinct Houtside].
  split; [ exact Hdistinct | ]. intros name Hname.
  apply Houtside, among_app_false in Hname as [Havoid Hvariables].
  apply among_app_false. split; [ exact Hvariables | ].
  destruct (among name free) eqn:Hamong; [ | reflexivity ].
  apply (proj1 (all_among _ _) Hfree) in Hamong. congruence.
Qed.

Theorem inline_fresh : inline_fresh_statement.
Proof.
  intros ap called c r avoid Hkind Hinlinable Havoid.
  apply inline_at;
    [ exact Hkind | exact Hinlinable | apply fresh_names_length
    | apply fresh_for_code, Havoid ].
Qed.

Theorem inline_fresh_state : inline_fresh_state_statement.
Proof.
  intros ap called c r avoid vars conts statics hp rd Hkind Hcode Hinlinable
    Havoid.
  apply (inline_at_state ap called c r);
    [ exact Hkind | exact Hcode | exact Hinlinable | apply fresh_names_length
    | apply fresh_for_code, Havoid ].
Qed.

Theorem inline_at_fresh : inline_at_fresh_statement.
Proof.
  intros ap called c r Hkind Hinlinable.
  exists (code_variables c ++ free_variables_of_simples (ap_args ap)).
  intros fresh Hlength Houtside. apply inline_at; assumption.
Qed.

(* ======================================================================== *)
(* Two renamings that agree on the free names                               *)
(* ======================================================================== *)

Lemma alpha_renamings_agree (q q' : renaming) (e : expr) :
  code_closed e = true ->
  (∀ x : variable,
    var_free_in_expr x e -> rename_variable q x = rename_variable q' x) ->
  (∀ k : continuation,
    cont_free_in_expr k e ->
    rename_continuation q k = rename_continuation q' k) ->
  alpha_equivalent (rename_expr q e) (rename_expr q' e).
Proof.
  intros Hclosed Hvariables Hcontinuations.
  set (t := compose_renaming q' (inverse_renaming q)).
  assert (Hsame : rename_expr t (rename_expr q e) = rename_expr q' e).
  { unfold t. rewrite rename_expr_compose, rename_expr_inverse.
    reflexivity. }
  rewrite <- Hsame. apply Alpha_rename_by_relation.
  - rewrite code_closed_renamed. exact Hclosed.
  - intros x Hx. unfold t. rewrite rename_variable_compose.
    assert (Hfree :
      var_free_in_expr (rename_variable (inverse_renaming q) x) e).
    { apply (var_free_equivariant q).
      rewrite rename_variable_inverse_r. exact Hx. }
    rewrite <- (Hvariables _ Hfree). apply rename_variable_inverse_r.
  - intros k Hk. unfold t. rewrite rename_continuation_compose.
    assert (Hfree :
      cont_free_in_expr (rename_continuation (inverse_renaming q) k) e).
    { apply (cont_free_equivariant q).
      rewrite rename_continuation_inverse_r. exact Hk. }
    rewrite <- (Hcontinuations _ Hfree). apply rename_continuation_inverse_r.
Qed.

(* ======================================================================== *)
(* The [Let]s of the parameters                                             *)
(* ======================================================================== *)

Lemma rename_let_params (q : renaming) (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr),
    rename_expr q (let_params params kinds args body)
    = let_params (map (rename_variable q) params) kinds
        (map (rename_simple q) args) (rename_expr q body).
Proof.
  induction params as [ | param params IH]; intros kinds [ | arg args] body;
    cbn [let_params map rename_expr rename_named]; try reflexivity.
  rewrite IH. reflexivity.
Qed.

Lemma let_params_code_closed (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr),
    code_closed (let_params params kinds args body) = code_closed body.
Proof.
  induction params as [ | param params IH]; intros kinds [ | arg args] body;
    cbn [let_params code_closed]; try reflexivity.
  apply IH.
Qed.

Fixpoint let_params_context (params : list variable)
    (kinds : list flambda_kind) (args : list simple) : context :=
  match params, args with
  | param :: params, arg :: args =>
    Let_body param (hd Value kinds) (Simple arg)
      (let_params_context params (tl kinds) args)
  | _, _ => Hole
  end.

Lemma plug_let_params (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr),
    plug (let_params_context params kinds args) body
    = let_params params kinds args body.
Proof.
  induction params as [ | param params IH]; intros kinds [ | arg args] body;
    cbn [let_params_context let_params plug]; try reflexivity.
  rewrite IH. reflexivity.
Qed.

(* A free variable of the [Let]s is free in an argument, or free in the
   body and not a parameter. *)
Lemma let_params_free_variables (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr)
    (x : variable),
    length params = length args ->
    among x (free_variables_of_expr (let_params params kinds args body))
    = true ->
    among x (free_variables_of_simples args) = true
    ∨ (among x (free_variables_of_expr body) = true
       ∧ among x params = false).
Proof.
  induction params as [ | param params IH];
    intros kinds [ | arg args] body x Hlength Hx;
    cbn [length] in Hlength; try discriminate Hlength.
  - right. split; [ exact Hx | reflexivity ].
  - injection Hlength as Hlength.
    cbn [let_params free_variables_of_expr free_variables_of_named] in Hx.
    rewrite among_app, among_without in Hx.
    unfold free_variables_of_simples. cbn [flat_map]. rewrite among_app.
    apply orb_true_iff in Hx as [Hx | Hx]; [ left; rewrite Hx; reflexivity | ].
    apply andb_true_iff in Hx as [Hx Hparam]. apply negb_true_iff in Hparam.
    rewrite among_cons in Hparam.
    apply orb_false_iff in Hparam as [Hparam _].
    destruct (IH (tl kinds) args body x Hlength Hx)
      as [Hargs | [Hbody Hparams]].
    + left. unfold free_variables_of_simples in Hargs. rewrite Hargs.
      apply orb_true_r.
    + right. split; [ exact Hbody | ].
      rewrite among_cons, Hparam, Hparams. reflexivity.
Qed.

(* ======================================================================== *)
(* The choice of the fresh variables                                        *)
(* ======================================================================== *)

Section Choice.

Variables (ap : apply_expr) (c : code) (r : renaming).

Hypothesis Hclosed : code_is_closed c = true.

Hypothesis Hdistinct : distinct (code_variables c) = true.

Hypothesis Hvars : ∀ x : variable, rename_variable r x = x.

Hypothesis Harity : length (code_params c) = length (ap_args ap).

(* Variables that will do. *)
Definition fresh_for_call (zs : list variable) : Prop :=
  length zs = length (code_variables c)
  ∧ distinct_outside (code_variables c ++ free_variables_of_apply ap) zs.

(* The body of the code, opened and with its continuations renamed. *)
Definition opened_body (zs : list variable) : expr :=
  rename_expr r
    (rename_expr (swap_variables (code_variables c) zs) (code_body c)).

Lemma inline_at_shape (zs : list variable) :
  fresh_for_call zs ->
  ∃ (z : variable) (rest : list variable),
    zs = z :: rest
    ∧ length rest = length (ap_args ap)
    ∧ inlined_at zs c r ap
      = Let z Value (Simple (ap_callee ap))
          (let_params rest (code_params_kinds c) (ap_args ap)
             (opened_body zs)).
Proof.
  intros [Hlength Houtside].
  apply distinct_outside_app in Houtside as [Houtside _].
  pose proof (open_code_variables zs c Hlength Hdistinct Houtside)
    as Hvariables.
  destruct zs as [ | z rest]; [ discriminate Hlength | ].
  exists z, rest. split; [ reflexivity | ].
  split;
    [ injection Hlength as Hlength; exact (eq_trans Hlength Harity) | ].
  unfold inlined_at, inlined_body, opened_body.
  rewrite open_code_body, open_code_kinds.
  unfold code_variables in Hvariables at 1.
  injection Hvariables as Hmy_closure Hparams.
  exact
    (f_equal2
       (fun my_closure params =>
          Let my_closure Value (Simple (ap_callee ap))
            (let_params params (code_params_kinds c) (ap_args ap)
               (rename_expr r
                  (rename_expr (swap_variables (code_variables c) (z :: rest))
                     (code_body c)))))
       Hmy_closure Hparams).
Qed.

Lemma closed_code_body : code_closed (code_body c) = true.
Proof. exact (proj2 (proj1 (closed_code c) Hclosed)). Qed.

Lemma opened_body_closed (zs : list variable) :
  code_closed (opened_body zs) = true.
Proof.
  unfold opened_body. rewrite !code_closed_renamed. exact closed_code_body.
Qed.

(* The free variables of the opened body are fresh ones. *)
Lemma opened_body_free_variables (zs : list variable) (x : variable) :
  fresh_for_call zs ->
  among x (free_variables_of_expr (opened_body zs)) = true ->
  among x zs = true.
Proof.
  intros [Hlength Houtside] Hx.
  apply distinct_outside_app in Houtside as [Houtside _].
  unfold opened_body in Hx. rewrite !free_variables_of_expr_renamed in Hx.
  rewrite (map_ext (rename_variable r) (fun y => y) Hvars), map_id in Hx.
  apply among_In, in_map_iff in Hx as (y & Hsame & Hy).
  apply among_In, (closed_code_variables c y Hclosed), among_In in Hy.
  pose proof
    (rename_variables_swap (code_variables c) zs Hlength Hdistinct Houtside)
    as Hswap.
  apply among_In.
  exact (eq_ind _ (fun names => x ∈ names)
           (eq_ind _ (fun name => name ∈ _) (in_map _ _ _ Hy) _ Hsame)
           _ Hswap).
Qed.

(* The swap of one list of fresh variables with another fixes the free
   variables of the call. *)
Lemma swap_fixes_call (zs ws : list variable) (x : variable) :
  fresh_for_call zs ->
  fresh_for_call ws ->
  among x (free_variables_of_apply ap) = true ->
  rename_variable (swap_variables zs ws) x = x.
Proof.
  intros [_ [_ Houtside]] [_ [_ Houtside']] Hx.
  apply rename_variables_swap_other.
  - destruct (among x zs) eqn:Hamong; [ | reflexivity ].
    apply Houtside, among_app_false in Hamong as [_ Hamong]. congruence.
  - destruct (among x ws) eqn:Hamong; [ | reflexivity ].
    apply Houtside', among_app_false in Hamong as [_ Hamong]. congruence.
Qed.

(* Opening at [zs] and then exchanging [zs] for [ws] is opening at
   [ws], on the variables of the code. *)
Lemma swaps_agree (zs ws : list variable) (x : variable) :
  fresh_for_call zs ->
  fresh_for_call ws ->
  distinct_outside zs ws ->
  among x (code_variables c) = true ->
  rename_variable (swap_variables zs ws)
    (rename_variable (swap_variables (code_variables c) zs) x)
  = rename_variable (swap_variables (code_variables c) ws) x.
Proof.
  intros [Hlength Houtside] [Hlength' Houtside'] Hdisjoint Hx.
  apply distinct_outside_app in Houtside as [Houtside _].
  apply distinct_outside_app in Houtside' as [Houtside' _].
  pose proof
    (rename_variables_swap (code_variables c) zs Hlength Hdistinct Houtside)
    as Hfirst.
  pose proof
    (rename_variables_swap zs ws (eq_trans Hlength' (eq_sym Hlength))
       (proj1 Houtside) Hdisjoint)
    as Hsecond.
  pose proof
    (rename_variables_swap (code_variables c) ws Hlength' Hdistinct Houtside')
    as Hboth.
  assert (Hall :
    map
      (fun y =>
         rename_variable (swap_variables zs ws)
           (rename_variable (swap_variables (code_variables c) zs) y))
      (code_variables c)
    = map (rename_variable (swap_variables (code_variables c) ws))
        (code_variables c)).
  { rewrite <- (map_map
                  (rename_variable (swap_variables (code_variables c) zs))
                  (rename_variable (swap_variables zs ws))).
    exact (eq_trans
             (eq_trans (f_equal (map _) Hfirst) Hsecond) (eq_sym Hboth)). }
  apply among_In in Hx.
  exact (proj1 map_ext_in_iff Hall x Hx).
Qed.

(* Two lists of fresh variables with no variable in common. *)
Lemma inline_at_moved (zs ws : list variable) :
  fresh_for_call zs ->
  fresh_for_call ws ->
  distinct_outside zs ws ->
  alpha_equivalent (inlined_at zs c r ap) (inlined_at ws c r ap).
Proof.
  intros Hzs Hws Hdisjoint.
  destruct (inline_at_shape zs Hzs) as (z & rest & Hzs_shape & Hcount & ->).
  destruct (inline_at_shape ws Hws) as (w & rest' & Hws_shape & Hcount' & ->).
  set (p := swap_variables zs ws).
  set (first :=
    Let z Value (Simple (ap_callee ap))
      (let_params rest (code_params_kinds c) (ap_args ap) (opened_body zs))).
  assert (Hclosed_first : code_closed first = true).
  { unfold first. cbn [code_closed].
    rewrite let_params_code_closed. apply opened_body_closed. }
  (* The free variables of the first term are those of the call. *)
  assert (Hfree :
    ∀ x : variable,
      var_free_in_expr x first ->
      among x (free_variables_of_apply ap) = true).
  { intros x Hx. apply (var_free_agrees x _ Hclosed_first) in Hx.
    unfold first in Hx.
    cbn [free_variables_of_expr free_variables_of_named] in Hx.
    unfold free_variables_of_apply. rewrite !among_app.
    rewrite among_app, among_without in Hx.
    apply orb_true_iff in Hx as [Hx | Hx]; [ rewrite Hx; reflexivity | ].
    apply andb_true_iff in Hx as [Hx Hnot]. apply negb_true_iff in Hnot.
    rewrite among_cons in Hnot. apply orb_false_iff in Hnot as [Hnot _].
    destruct
      (let_params_free_variables rest (code_params_kinds c) (ap_args ap)
         (opened_body zs) x Hcount Hx)
      as [Hargs | [Hbody Hrest]];
      [ rewrite Hargs; cbn [orb]; apply orb_true_r | ].
    apply (opened_body_free_variables zs x Hzs) in Hbody.
    rewrite Hzs_shape, among_cons, Hnot, Hrest in Hbody.
    discriminate Hbody. }
  apply Alpha_trans with (rename_expr p first).
  { apply (Alpha_rename_by_relation p first Hclosed_first).
    - intros x Hx. exact (swap_fixes_call zs ws x Hzs Hws (Hfree x Hx)).
    - intros k _. reflexivity. }
  (* The first term renamed is the second, up to the body. *)
  assert (Hcallee : rename_simple p (ap_callee ap) = ap_callee ap).
  { apply rename_simple_fixes. intros x Hx.
    apply (swap_fixes_call zs ws x Hzs Hws).
    unfold free_variables_of_apply. rewrite among_app, Hx. reflexivity. }
  assert (Hargs : map (rename_simple p) (ap_args ap) = ap_args ap).
  { apply rename_simples_fixes. intros x Hx.
    apply (swap_fixes_call zs ws x Hzs Hws).
    unfold free_variables_of_apply. rewrite !among_app, Hx. cbn [orb].
    apply orb_true_r. }
  assert (Hmoved : map (rename_variable p) (z :: rest) = w :: rest').
  { destruct Hzs as [Hlength [Hdistinct_zs _]]. destruct Hws as [Hlength' _].
    pose proof
      (rename_variables_swap zs ws (eq_trans Hlength' (eq_sym Hlength))
         Hdistinct_zs Hdisjoint)
      as Hswap.
    exact (eq_trans
             (f_equal (map (rename_variable p)) (eq_sym Hzs_shape))
             (eq_trans Hswap Hws_shape)). }
  cbn [map] in Hmoved. injection Hmoved as Hfirst_moved Hrest_moved.
  assert (Hrenamed :
    rename_expr p first
    = Let w Value (Simple (ap_callee ap))
        (let_params rest' (code_params_kinds c) (ap_args ap)
           (rename_expr p (opened_body zs)))).
  { unfold first. cbn [rename_expr rename_named].
    rewrite rename_let_params, Hcallee, Hargs.
    exact
      (f_equal2
         (fun bound params =>
            Let bound Value (Simple (ap_callee ap))
              (let_params params (code_params_kinds c) (ap_args ap)
                 (rename_expr p (opened_body zs))))
         Hfirst_moved Hrest_moved). }
  rewrite Hrenamed.
  (* The bodies. *)
  assert (Hbodies :
    alpha_equivalent (rename_expr p (opened_body zs)) (opened_body ws)).
  { unfold opened_body.
    rewrite <- (rename_expr_compose r), <- (rename_expr_compose p),
      <- (rename_expr_compose r).
    apply alpha_renamings_agree; [ exact closed_code_body | | ].
    - intros x Hx. rewrite !rename_variable_compose, !Hvars.
      apply (swaps_agree zs ws x Hzs Hws Hdisjoint).
      apply (closed_code_variables c x Hclosed).
      apply (var_free_agrees x _ closed_code_body), Hx.
    - intros k _. rewrite !rename_continuation_compose. reflexivity. }
  pose proof
    (Alpha_context
       (Let_body w Value (Simple (ap_callee ap))
          (let_params_context rest' (code_params_kinds c) (ap_args ap)))
       _ _ Hbodies)
    as Hcontext.
  cbn [plug] in Hcontext. rewrite !plug_let_params in Hcontext.
  exact Hcontext.
Qed.

End Choice.

Theorem inline_at_choice : inline_at_choice_statement.
Proof.
  intros ap c r fresh fresh' Hclosed Hdistinct Hvars Harity Hlength Hlength'
    Houtside Houtside'.
  set (avoid :=
    (code_variables c ++ free_variables_of_apply ap) ++ fresh ++ fresh').
  set (third := fresh_names (length (code_variables c)) avoid).
  pose proof (fresh_names_length (length (code_variables c)) avoid)
    as Hcount.
  pose proof (fresh_names_outside (length (code_variables c)) avoid)
    as Hthird.
  apply distinct_outside_app in Hthird as [Hthird Hdisjoint].
  apply distinct_outside_app in Hdisjoint as [Hdisjoint Hdisjoint'].
  apply Alpha_trans with (inlined_at third c r ap).
  - apply inline_at_moved; try assumption; split; assumption.
  - apply Alpha_sym, inline_at_moved; try assumption; split; assumption.
Qed.

Theorem inline_fresh_choice : inline_fresh_choice_statement.
Proof.
  intros ap c r avoid avoid' Hclosed Hdistinct Hvars Harity Havoid Havoid'.
  apply inline_at_choice; try assumption;
    solve [ apply fresh_names_length | apply fresh_for_code; assumption ].
Qed.
