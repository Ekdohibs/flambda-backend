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

(* Proofs of the statements of FlambdaLiftContsSpec.v. Nothing here
   needs review. That the pass commutes with renaming is the lemma of
   FlambdaLiftContsNamesProofs.v; what follows is of the equivalence.

   One continuation moved out of one handler first, in two steps that
   are each an equivalence; then the continuations of a handler in
   turn; then the pass, by the context lemmas of FlambdaContextProofs.v.

   The first step gives the continuation the parameters that it reads
   and leaves it where it is. It holds in the states where those
   parameters have values of their kinds, which the handler around it
   is entered in. It is made one parameter at a time, and each is the
   equivalence of FlambdaUnaliasParamsProofs.v read from right to
   left: the continuation with the parameter [p] more, given [p] by
   every application, is the continuation whose handler begins with a
   [Let] of [p]. That equivalence is stated of a parameter whose name
   is not the variable given, so the parameter is first given another
   name, which is made here and is in no term that the pass returns.

   The second step moves the continuation, whose handler no longer
   reads a parameter of the handler around it, out of that handler:
   FlambdaSinkContsProofs.let_cont_into_handler, the other way. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaPrimitiveProofs FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaUnusedContsSpec FlambdaUnusedContsProofs FlambdaContAlikeProofs
  FlambdaSinkContsProofs FlambdaBindersSpec FlambdaFreeNamesSpec
  FlambdaFreeNamesProofs FlambdaNameAbstractionSpec
  FlambdaNameAbstractionProofs FlambdaUnaliasParamsSpec
  FlambdaUnaliasParamsProofs FlambdaLiftContsSpec.
From OxCamlFlambda2 Require FlambdaLiftContsNamesProofs.
Import ListNotations.

(* ======================================================================== *)
(* Giving arguments, and taking one away                                    *)
(* ======================================================================== *)

Lemma give_to_nil (j : continuation) (ac : apply_cont) :
  give_to j [] ac = ac.
Proof.
  unfold give_to. destruct (Pos.eqb (ac_continuation ac) j); [ | reflexivity ].
  destruct ac; reflexivity.
Qed.

Fixpoint give_nil (j : continuation) (e : expr) {struct e} :
  give j [] e = e
with give_in_handler_nil (j : continuation) (h : continuation_handler)
    {struct h} :
  give_in_handler j [] h = h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [give]; try reflexivity.
    + rewrite give_nil. reflexivity.
    + rewrite give_nil. reflexivity.
    + rewrite give_in_handler_nil, give_nil. reflexivity.
    + rewrite give_nil. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map]. rewrite give_in_handler_nil, IH. reflexivity.
    + rewrite give_to_nil. reflexivity.
    + f_equal. induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map]. rewrite give_to_nil, IH. reflexivity.
  - destruct h as [params metadata handler]. cbn [give_in_handler].
    rewrite give_nil. reflexivity.
Qed.

(* The variables given one at a time, the last first. *)
Lemma give_to_cons (j : continuation) (p : variable) (ps : list variable)
    (ac : apply_cont) :
  give_to j (p :: ps) ac = give_to j [p] (give_to j ps ac).
Proof.
  unfold give_to.
  destruct (Pos.eqb (ac_continuation ac) j) eqn:Hj; cbn [ac_continuation];
    rewrite Hj; reflexivity.
Qed.

Fixpoint give_cons (j : continuation) (p : variable) (ps : list variable)
    (e : expr) {struct e} :
  give j (p :: ps) e = give j [p] (give j ps e)
with give_in_handler_cons (j : continuation) (p : variable)
    (ps : list variable) (h : continuation_handler) {struct h} :
  give_in_handler j (p :: ps) h
  = give_in_handler j [p] (give_in_handler j ps h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [give]; try reflexivity.
    + rewrite give_cons. reflexivity.
    + rewrite give_cons. reflexivity.
    + rewrite give_in_handler_cons, give_cons. reflexivity.
    + rewrite give_cons. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map]. rewrite give_in_handler_cons, IH. reflexivity.
    + rewrite give_to_cons. reflexivity.
    + f_equal. induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map]. rewrite give_to_cons, IH. reflexivity.
  - destruct h as [params metadata handler]. cbn [give_in_handler].
    rewrite give_cons. reflexivity.
Qed.

(* The argument given, taken away again. *)
Lemma drop_arg_of_give_to (j : continuation) (p : variable) (ac : apply_cont) :
  drop_arg_of j 0 (give_to j [p] ac) = ac.
Proof.
  unfold drop_arg_of, give_to.
  destruct (Pos.eqb (ac_continuation ac) j) eqn:Hj; cbn [ac_continuation];
    rewrite Hj; [ | reflexivity ].
  destruct ac; reflexivity.
Qed.

Fixpoint drop_arg_give (j : continuation) (p : variable) (e : expr)
    {struct e} :
  drop_arg j 0 (give j [p] e) = e
with drop_arg_give_in_handler (j : continuation) (p : variable)
    (h : continuation_handler) {struct h} :
  drop_arg_in_handler j 0 (give_in_handler j [p] h) = h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [give drop_arg]; try reflexivity.
    + rewrite drop_arg_give. reflexivity.
    + rewrite drop_arg_give. reflexivity.
    + rewrite drop_arg_give_in_handler, drop_arg_give. reflexivity.
    + rewrite drop_arg_give. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map]. rewrite drop_arg_give_in_handler, IH. reflexivity.
    + rewrite drop_arg_of_give_to. reflexivity.
    + f_equal. induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map]. rewrite drop_arg_of_give_to, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [give_in_handler drop_arg_in_handler].
    rewrite drop_arg_give. reflexivity.
Qed.

(* ======================================================================== *)
(* What the test of the pass gives the test of the other                    *)
(* ======================================================================== *)

(* Where the pass finds every mention of [j] an application, with none
   of the variables that [j] is to take bound on the way, the term in
   which [j] is given [p] first passes the test of
   FlambdaUnaliasParamsSpec.v for the first argument and [p]. *)
Lemma applies_only_gives_var (j : continuation) (p : variable)
    (ps : list variable) (ok ok' : bool) (ac : apply_cont) :
  (ok = true -> ok' = true) ->
  applies_only j ok ac = true ->
  apply_cont_gives_var j 0 p ok' (give_to j (p :: ps) ac) = true.
Proof.
  intros Hok. unfold applies_only, apply_cont_gives_var, give_to.
  intros Hac. apply andb_true_iff in Hac as [Htrap Happlied].
  destruct (Pos.eqb (ac_continuation ac) j) eqn:Hj;
    cbn [ac_continuation ac_trap_action ac_args]; rewrite Hj, Htrap;
    [ | reflexivity ].
  cbn [map app nth_error]. rewrite Pos.eqb_refl, andb_true_r.
  apply Hok. rewrite orb_false_r in Happlied. exact Happlied.
Qed.

Fixpoint only_applied_all_give_var (j : continuation) (p : variable)
    (ps ps' : list variable) (e : expr) {struct e} :
  ∀ ok ok' : bool,
    (ok = true -> ok' = true) ->
    p ∈ ps ->
    only_applied j ps ok e = true ->
    all_give_var j 0 p ok' (give j (p :: ps') e) = true
with handler_only_applied_all_give_var (j : continuation) (p : variable)
    (ps ps' : list variable) (h : continuation_handler) {struct h} :
  ∀ ok ok' : bool,
    (ok = true -> ok' = true) ->
    p ∈ ps ->
    handler_only_applied j ps ok h = true ->
    handler_all_give_var j 0 p ok' (give_in_handler j (p :: ps') h) = true.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros ok ok' Hok Hp; cbn [only_applied give all_give_var];
      try (intros H; exact H).
    + apply only_applied_all_give_var; [ | exact Hp ].
      intros Hboth. apply andb_true_iff in Hboth as [Hok1 Hy].
      rewrite (Hok Hok1). cbn [andb]. apply negb_true_iff.
      apply negb_true_iff in Hy. apply Pos.eqb_neq. intros <-.
      apply not_among_in in Hy. contradiction.
    + apply only_applied_all_give_var; assumption.
    + intros H. apply andb_true_iff in H as [H Hbody].
      apply andb_true_iff in H as [Hne Hh].
      rewrite Hne,
        (handler_only_applied_all_give_var j p ps ps' hh ok ok' Hok Hp Hh),
        (only_applied_all_give_var j p ps ps' body ok ok' Hok Hp Hbody).
      reflexivity.
    + intros H. apply andb_true_iff in H as [Hhandlers Hbody].
      rewrite (only_applied_all_give_var j p ps ps' body ok ok' Hok Hp Hbody),
        andb_true_r.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb] in Hhandlers |- *.
      apply andb_true_iff in Hhandlers as [Hfirst Hothers].
      apply andb_true_iff in Hfirst as [Hne Hh].
      rewrite Hne,
        (handler_only_applied_all_give_var j p ps ps' hh ok ok' Hok Hp Hh),
        (IH Hothers).
      reflexivity.
    + apply applies_only_gives_var, Hok.
    + intros Harms.
      induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map forallb] in Harms |- *.
      apply andb_true_iff in Harms as [Hac Harms].
      rewrite (applies_only_gives_var j p ps' ok ok' ac Hok Hac), (IH Harms).
      reflexivity.
  - destruct h as [params metadata handler]. intros ok ok' Hok Hp.
    cbn [handler_only_applied give_in_handler handler_all_give_var].
    apply only_applied_all_give_var; [ | exact Hp ].
    intros Hboth. apply andb_true_iff in Hboth as [Hok1 Hparams].
    rewrite (Hok Hok1). cbn [andb]. apply negb_true_iff.
    apply negb_true_iff in Hparams.
    destruct (among p params) eqn:Hamong; [ | reflexivity ].
    assert (Hex : existsb (fun p0 => among p0 params) ps = true).
    { apply existsb_exists. exists p. split; assumption. }
    congruence.
Qed.

(* ======================================================================== *)
(* One parameter more                                                       *)
(* ======================================================================== *)

(* A variable has a value of a kind. *)
Definition has_kind (vars : env) (p : variable) (kind : flambda_kind) : Prop :=
  ∃ v : value, lookup p vars = Some v ∧ kind_of_value v = kind.

(* FlambdaUnaliasParamsProofs.unalias_param_equivalent at the first
   position, of a variable that is known without a search of the
   applications. *)
Lemma unalias_first_equivalent (j : continuation) (x p : variable)
    (kind : flambda_kind) (q : list variable) (kinds : list flambda_kind)
    (handler body : expr) :
  among x (p :: q) = false ->
  among p q = false ->
  all_give_var j 0 x true body = true ->
  expr_equivalent
    (Let_cont_non_recursive j
       (Handler (p :: q)
          {| hm_params_kinds := kind :: kinds; hm_is_exn_handler := false |}
          handler)
       body)
    (Let_cont_non_recursive j
       (Handler q
          {| hm_params_kinds := kinds; hm_is_exn_handler := false |}
          (Let p kind (Simple (Var x)) handler))
       (drop_arg j 0 body)).
Proof.
  intros Hx Hfresh Hgive vars conts statics hp.
  set (metadata :=
    {| hm_params_kinds := kind :: kinds; hm_is_exn_handler := false |}).
  apply (equivalent_trans _
    (State body vars
       ((j, the_closure (p :: q) metadata handler vars conts) :: conts)
       statics hp));
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply (equivalent_trans _
    (State (drop_arg j 0 body) vars
       ((j, the_closure' 0 x p kind (p :: q) metadata handler vars conts)
        :: conts)
       statics hp));
    [ | apply equivalent_sym, step_equivalent, step_let_cont_non_recursive ].
  apply
    (site_equivalent j 0 x p kind (p :: q) metadata handler vars conts
       eq_refl eq_refl Hx Hfresh).
  apply (Rel_state_inside j 0 x p kind (p :: q) metadata handler vars conts
    true);
    [ apply agree_refl | reflexivity | exact Hgive | ].
  split; [ apply resolve_head | split; [ apply resolve_head | ] ].
  intros k1 Hne. rewrite !(resolve_other k1 j) by exact Hne. apply Rel_same.
Qed.

(* A handler that begins by binding a variable to itself through
   another name, where the variable has a value of the kind and is not
   a parameter, applies as the handler without that. *)
Lemma rebound_alike (p p' : variable) (kind : flambda_kind)
    (q : list variable) (metadata : handler_metadata) (h : expr) (vars : env)
    (conts : cont_env) :
  among p q = false ->
  ~ var_free_in_expr p' h ->
  has_kind vars p kind ->
  applies_alike (Closure (Handler q metadata h) vars conts [])
    (Closure
       (Handler q metadata
          (Let p' kind (Simple (Var p)) (Let p kind (Simple (Var p')) h)))
       vars conts []).
Proof.
  intros Hp Hp' [v [Hv Hkind]] vs statics hp. simpl.
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
    simpl; [ | exact I ].
  destruct (bind_params q vs vars) as [bound | ] eqn:Hbind; simpl;
    [ | exact I ].
  assert (Hbound : lookup p bound = Some v).
  { rewrite (lookup_bind_params_out q vs vars bound p Hp Hbind). exact Hv. }
  apply equivalent_sym.
  eapply equivalent_trans.
  { apply step_equivalent. rewrite step_let. cbn [eval_named eval_simple].
    rewrite Hbound, Hkind, kind_eqb_refl. reflexivity. }
  eapply equivalent_trans.
  { apply step_equivalent. rewrite step_let. cbn [eval_named eval_simple].
    rewrite lookup_head, Hkind, kind_eqb_refl. reflexivity. }
  apply coincide_equivalent;
    [ | intros k _; unfold conts_agree_at; apply Same_matches ].
  intros y Hy. rewrite lookup_cons.
  destruct (Pos.eqb_spec y p) as [-> | _]; [ exact Hbound | ].
  symmetry. apply lookup_other. intros ->. contradiction.
Qed.

(* A handler whose first parameter has another name, and that begins by
   binding the variable to it, applies as the handler whose first
   parameter is the variable. *)
Lemma renamed_param_alike (p p' : variable) (kind : flambda_kind)
    (q : list variable) (kinds : list flambda_kind) (h : expr) (vars : env)
    (conts : cont_env) :
  among p q = false ->
  among p' q = false ->
  ~ var_free_in_expr p' h ->
  applies_alike
    (Closure
       (Handler (p' :: q)
          {| hm_params_kinds := kind :: kinds; hm_is_exn_handler := false |}
          (Let p kind (Simple (Var p')) h))
       vars conts [])
    (Closure
       (Handler (p :: q)
          {| hm_params_kinds := kind :: kinds; hm_is_exn_handler := false |}
          h)
       vars conts []).
Proof.
  intros Hp Hp' Hfree vs statics hp. simpl.
  destruct (kinds_eqb (map kind_of_value vs) (kind :: kinds)) eqn:Hkinds;
    simpl; [ | exact I ].
  destruct vs as [ | v vs]; simpl; [ exact I | ].
  apply kinds_eqb_eq in Hkinds. cbn [map] in Hkinds.
  injection Hkinds as Hkind _.
  pose proof (bind_params_under q vs vars p' v Hp') as Hunder'.
  pose proof (bind_params_under q vs vars p v Hp) as Hunder.
  destruct (bind_params q vs ((p', v) :: vars)) as [bound' | ],
    (bind_params q vs ((p, v) :: vars)) as [bound | ],
    (bind_params q vs vars) as [bound0 | ];
    try contradiction; simpl; [ | exact I ].
  eapply equivalent_trans.
  { apply step_equivalent. rewrite step_let. cbn [eval_named eval_simple].
    rewrite Hunder', lookup_head, Hkind, kind_eqb_refl. reflexivity. }
  apply coincide_equivalent;
    [ | intros k _; unfold conts_agree_at; apply Same_matches ].
  intros y Hy. rewrite Hunder, !lookup_cons.
  destruct (Pos.eqb_spec y p) as [-> | _]; [ reflexivity | ].
  rewrite Hunder'. symmetry. apply lookup_other. intros ->. contradiction.
Qed.

(* A continuation given, as its first parameter, a variable that has a
   value of the kind where the continuation is bound, by applications
   at which the variable is still that. *)
Lemma give_one_equivalent (j : continuation) (p : variable)
    (kind : flambda_kind) (q : list variable) (kinds : list flambda_kind)
    (h rest : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) :
  among p q = false ->
  all_give_var j 0 p true (give j [p] rest) = true ->
  has_kind vars p kind ->
  equivalent
    (State
       (Let_cont_non_recursive j
          (Handler q
             {| hm_params_kinds := kinds; hm_is_exn_handler := false |} h)
          rest)
       vars conts statics hp)
    (State
       (Let_cont_non_recursive j
          (Handler (p :: q)
             {| hm_params_kinds := kind :: kinds;
                hm_is_exn_handler := false |}
             h)
          (give j [p] rest))
       vars conts statics hp).
Proof.
  intros Hp Hgive Hkind.
  set (p' := fresh_name (p :: q ++ free_variables_of_expr h)).
  pose proof (fresh_name_fresh (p :: q ++ free_variables_of_expr h)) as Hfresh.
  fold p' in Hfresh. rewrite among_cons, among_app in Hfresh.
  apply orb_false_iff in Hfresh as [Hne Hfresh].
  apply orb_false_iff in Hfresh as [Hq Hfree].
  apply Pos.eqb_neq in Hne.
  assert (Hnot : ~ var_free_in_expr p' h).
  { intros Hin. apply var_free_among in Hin. congruence. }
  eapply equivalent_trans.
  { apply let_cont_handler_alike.
    exact (rebound_alike p p' kind q _ h vars conts Hp Hnot Hkind). }
  eapply equivalent_trans.
  { apply equivalent_sym.
    pose proof
      (unalias_first_equivalent j p p' kind q kinds
         (Let p kind (Simple (Var p')) h) (give j [p] rest))
      as Hunalias.
    rewrite drop_arg_give in Hunalias. apply Hunalias.
    - rewrite among_cons, Hp, orb_false_r. apply Pos.eqb_neq. congruence.
    - exact Hq.
    - exact Hgive. }
  apply let_cont_handler_alike.
  exact (renamed_param_alike p p' kind q kinds h vars conts Hp Hq Hnot).
Qed.

(* ======================================================================== *)
(* The parameters that a handler reads                                      *)
(* ======================================================================== *)

(* Each of them given, the last first. *)
Lemma give_params_equivalent (j : continuation) (all q : list variable)
    (kinds : list flambda_kind) (h rest : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) :
  only_applied j all true rest = true ->
  ∀ read : list (variable * flambda_kind),
    (∀ p : variable, p ∈ map fst read -> p ∈ all) ->
    distinct (map fst read ++ q) = true ->
    Forall (fun '(p, kind) => has_kind vars p kind) read ->
    equivalent
      (State
         (Let_cont_non_recursive j
            (Handler q
               {| hm_params_kinds := kinds; hm_is_exn_handler := false |} h)
            rest)
         vars conts statics hp)
      (State
         (Let_cont_non_recursive j
            (Handler (map fst read ++ q)
               {| hm_params_kinds := map snd read ++ kinds;
                  hm_is_exn_handler := false |}
               h)
            (give j (map fst read) rest))
         vars conts statics hp).
Proof.
  intros Honly read.
  induction read as [ | [p kind] read IH]; intros Hall Hdistinct Hkinds.
  - cbn [map app]. rewrite give_nil. apply equivalent_refl.
  - cbn [map fst snd app] in Hall, Hdistinct |- *.
    cbn [distinct] in Hdistinct.
    apply andb_true_iff in Hdistinct as [Hp Hdistinct].
    apply negb_true_iff in Hp.
    apply Forall_cons_iff in Hkinds as [Hkind Hkinds].
    eapply equivalent_trans.
    { apply IH; [ | exact Hdistinct | exact Hkinds ].
      intros p0 Hin. apply Hall. right. exact Hin. }
    rewrite give_cons. apply give_one_equivalent; [ exact Hp | | exact Hkind ].
    rewrite <- give_cons.
    apply (only_applied_all_give_var j p all (map fst read) rest true true);
      [ auto | | exact Honly ].
    apply Hall. left. reflexivity.
Qed.

(* The parameters that the pass finds read are among the parameters. *)
Lemma among_read_params (x : variable) (e : expr) (params : list variable) :
  ∀ kinds : list flambda_kind,
    among x (map fst (read_params params kinds e)) = true ->
    among x params = true.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds];
    cbn [read_params map]; try discriminate.
  rewrite among_cons.
  destruct (var_free_b p e); cbn [map fst]; [ rewrite among_cons | ];
    intros Hx.
  - apply orb_true_iff in Hx as [Hx | Hx]; [ rewrite Hx; reflexivity | ].
    rewrite (IH kinds Hx). apply orb_true_r.
  - rewrite (IH kinds Hx). apply orb_true_r.
Qed.

(* They are distinct from each other and from the names that the
   parameters are distinct from. *)
Lemma distinct_read_params (e : expr) (q : list variable)
    (params : list variable) :
  ∀ kinds : list flambda_kind,
    distinct (params ++ q) = true ->
    distinct (map fst (read_params params kinds e) ++ q) = true.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds];
    cbn [read_params map app distinct]; try (intros H; exact H).
  - intros H. apply andb_true_iff in H as [_ H].
    clear IH. induction params as [ | p' params IH']; [ exact H | ].
    cbn [app distinct] in H. apply andb_true_iff in H as [_ H]. exact (IH' H).
  - intros H. apply andb_true_iff in H as [Hp H].
    specialize (IH kinds H).
    destruct (var_free_b p e); [ | exact IH ].
    cbn [map fst app distinct]. rewrite IH, andb_true_r.
    apply negb_true_iff. apply negb_true_iff in Hp.
    rewrite among_app in Hp |- *. apply orb_false_iff in Hp as [Hp Hq].
    rewrite Hq, orb_false_r.
    destruct (among p (map fst (read_params params kinds e))) eqn:Hread;
      [ | reflexivity ].
    apply among_read_params in Hread. congruence.
Qed.

(* A parameter that the test says is free is one of them, where every
   parameter has a kind. *)
Lemma in_read_params (x : variable) (e : expr) (params : list variable) :
  ∀ kinds : list flambda_kind,
    length kinds = length params ->
    x ∈ params ->
    var_free_b x e = true ->
    x ∈ map fst (read_params params kinds e).
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds];
    cbn [length read_params]; try discriminate; intros Hlength Hin Hfree;
    [ contradiction | ].
  injection Hlength as Hlength.
  destruct Hin as [-> | Hin].
  - rewrite Hfree. left. reflexivity.
  - specialize (IH kinds Hlength Hin Hfree).
    destruct (var_free_b p e); [ right | ]; exact IH.
Qed.

(* Where a handler is entered, each of them has a value of its kind. *)
Lemma bound_has_kinds (e : expr) (params : list variable) :
  ∀ (vs : list value) (vars bound : env),
    bind_params params vs vars = Some bound ->
    distinct params = true ->
    Forall (fun '(p, kind) => has_kind bound p kind)
      (read_params params (map kind_of_value vs) e).
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars bound;
    cbn [bind_params read_params map distinct]; try discriminate;
    intros Hbind Hdistinct; [ constructor | ].
  apply andb_true_iff in Hdistinct as [Hp Hdistinct].
  apply negb_true_iff in Hp.
  specialize (IH vs _ bound Hbind Hdistinct).
  destruct (var_free_b p e); [ | exact IH ].
  constructor; [ | exact IH ].
  exists v. split; [ | reflexivity ].
  rewrite (lookup_bind_params_out params vs _ bound p Hp Hbind).
  apply lookup_head.
Qed.

(* ======================================================================== *)
(* One continuation out of a handler                                        *)
(* ======================================================================== *)

(* It takes the parameters that it reads, where it is. *)
Lemma take_params_equivalent (k j : continuation) (params q : list variable)
    (metadata : handler_metadata) (kinds : list flambda_kind)
    (h rest body : expr) :
  distinct params = true ->
  distinct (params ++ q) = true ->
  only_applied j
    (map fst (read_params params (hm_params_kinds metadata) h)) true rest
  = true ->
  expr_equivalent
    (Let_cont_non_recursive k
       (Handler params metadata
          (Let_cont_non_recursive j
             (Handler q
                {| hm_params_kinds := kinds; hm_is_exn_handler := false |} h)
             rest))
       body)
    (Let_cont_non_recursive k
       (Handler params metadata
          (Let_cont_non_recursive j
             (Handler
                (map fst (read_params params (hm_params_kinds metadata) h)
                 ++ q)
                {| hm_params_kinds :=
                     map snd
                       (read_params params (hm_params_kinds metadata) h)
                     ++ kinds;
                   hm_is_exn_handler := false |}
                h)
             (give j
                (map fst (read_params params (hm_params_kinds metadata) h))
                rest)))
       body).
Proof.
  intros Hparams Hdistinct Honly vars conts statics hp.
  apply let_cont_handler_alike. intros vs statics' hp'. simpl.
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
    eqn:Hkinds; simpl; [ | exact I ].
  destruct (bind_params params vs vars) as [bound | ] eqn:Hbind; simpl;
    [ | exact I ].
  apply kinds_eqb_eq in Hkinds. rewrite <- Hkinds in Honly |- *.
  apply
    (give_params_equivalent j
       (map fst (read_params params (map kind_of_value vs) h)) q kinds h rest
       bound conts statics' hp' Honly);
    [ auto | | ].
  - apply distinct_read_params, Hdistinct.
  - exact (bound_has_kinds h params vs vars bound Hbind Hparams).
Qed.

(* It then reads no parameter of the handler around it, and goes out. *)
Lemma move_out_equivalent (k j : continuation) (params : list variable)
    (metadata : handler_metadata) (hj : continuation_handler)
    (rest body : expr) :
  ~ cont_free_in_expr j body ->
  ~ cont_free_in_handler j hj ->
  (∀ param : variable, param ∈ params -> ~ var_free_in_handler param hj) ->
  expr_equivalent
    (Let_cont_non_recursive k
       (Handler params metadata (Let_cont_non_recursive j hj rest)) body)
    (Let_cont_non_recursive j hj
       (Let_cont_non_recursive k (Handler params metadata rest) body)).
Proof.
  intros Hbody Hhj Hparams vars conts statics hp.
  apply equivalent_sym.
  exact (let_cont_into_handler j k hj params metadata rest body Hbody Hhj
    Hparams vars conts statics hp).
Qed.

(* ======================================================================== *)
(* The continuations of a handler in turn                                   *)
(* ======================================================================== *)

Lemma give_all_let_cont (moved : list (continuation * list variable))
    (j : continuation) (q : list variable) (metadata : handler_metadata)
    (h rest : expr) :
  give_all moved (Let_cont_non_recursive j (Handler q metadata h) rest)
  = Let_cont_non_recursive j (Handler q metadata (give_all moved h))
      (give_all moved rest).
Proof.
  unfold give_all.
  induction moved as [ | [j' ps] moved IH]; [ reflexivity | ].
  cbn [fold_right]. rewrite IH. reflexivity.
Qed.

Lemma lift_spine_equivalent (k : continuation) (params : list variable)
    (metadata : handler_metadata) (body : expr) :
  distinct params = true ->
  length (hm_params_kinds metadata) = length params ->
  ∀ (handler : expr) (moved : list (continuation * list variable))
    (lifted : list (continuation * continuation_handler)) (switch : expr),
    lift_spine body params (hm_params_kinds metadata) moved handler
    = Some (lifted, switch) ->
    expr_equivalent
      (Let_cont_non_recursive k
         (Handler params metadata (give_all moved handler)) body)
      (bind_lifted lifted
         (Let_cont_non_recursive k (Handler params metadata switch) body)).
Proof.
  intros Hparams Hlength handler.
  induction handler
    as [ y kind d rest _ | bindings rest _ | j [q mdj h] rest IH
       | handlers rest _ | ap | ac | scrutinee arms | ];
    intros moved lifted switch; cbn [lift_spine]; try discriminate.
  - (* a continuation of the handler *)
    set (h' := give_all moved h).
    set (read := read_params params (hm_params_kinds metadata) h').
    destruct
      (negb (hm_is_exn_handler mdj) && distinct (params ++ q)
       && negb (cont_free_b j body) && negb (cont_free_b j h')
       && only_applied j (map fst read) true (give_all moved rest))
      eqn:Htests; [ | discriminate ].
    apply andb_true_iff in Htests as [Htests Honly].
    apply andb_true_iff in Htests as [Htests Hh'].
    apply andb_true_iff in Htests as [Htests Hbody].
    apply andb_true_iff in Htests as [Hexn Hdistinct].
    apply negb_true_iff in Hexn, Hbody, Hh'.
    destruct
      (lift_spine body params (hm_params_kinds metadata)
         ((j, map fst read) :: moved) rest)
      as [[others switch'] | ] eqn:Hrest; [ | discriminate ].
    intros [= <- <-]. cbn [bind_lifted fold_right].
    rewrite give_all_let_cont. fold h'.
    destruct mdj as [kinds exn]. cbn [hm_is_exn_handler hm_params_kinds] in *.
    subst exn.
    eapply expr_equivalent_trans.
    { exact (take_params_equivalent k j params q metadata kinds h'
        (give_all moved rest) body Hparams Hdistinct Honly). }
    fold read.
    eapply expr_equivalent_trans.
    { apply move_out_equivalent.
      - intros Hfree. apply cont_free_b_sound in Hfree. congruence.
      - cbn [cont_free_in_handler]. intros Hfree.
        apply cont_free_b_sound in Hfree. congruence.
      - intros param Hin [Hnotin Hfree]. apply Hnotin, in_or_app. left.
        apply in_read_params; [ exact Hlength | exact Hin | ].
        apply var_free_b_sound, Hfree. }
    exact (plug_equivalent
      (Let_cont_non_recursive_body j
         (Handler (map fst read ++ q)
            {| hm_params_kinds := map snd read ++ kinds;
               hm_is_exn_handler := false |}
            h')
         Hole)
      _ _ (not_many _) (IH ((j, map fst read) :: moved) others switch' Hrest)).
  - (* the [Switch] *)
    destruct scrutinee as [p | sym | c]; try discriminate.
    destruct (among p params); [ | discriminate ].
    intros [= <- <-]. intros vars conts statics hp. apply equivalent_refl.
Qed.

Lemma lift_conts_equivalent (k : continuation) (h : continuation_handler)
    (body lifted : expr) :
  lift_conts k h body = Some lifted ->
  expr_equivalent (Let_cont_non_recursive k h body) lifted.
Proof.
  destruct h as [params metadata handler]. unfold lift_conts.
  destruct (tested params handler) as [i | ]; [ | discriminate ].
  destruct
    (negb (hm_is_exn_handler metadata) && distinct params
     && Nat.eqb (length (hm_params_kinds metadata)) (length params)
     && all_give_immediate k i body
     && negb (number_of_a_scrutinee k i body))
    eqn:Htests; [ | discriminate ].
  apply andb_true_iff in Htests as [Htests _].
  apply andb_true_iff in Htests as [Htests _].
  apply andb_true_iff in Htests as [Htests Hlength].
  apply andb_true_iff in Htests as [_ Hparams].
  apply Nat.eqb_eq in Hlength.
  destruct (lift_spine body params (hm_params_kinds metadata) [] handler)
    as [[[ | first others] switch] | ] eqn:Hspine; try discriminate.
  destruct (read_where_given k i params switch body); [ | discriminate ].
  intros [= <-].
  exact (lift_spine_equivalent k params metadata body Hparams Hlength handler
    [] (first :: others) switch Hspine).
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* Each layer is a context around what the induction gives; then the
   continuations of the handler are looked at. *)
Fixpoint lift_handler_conts_equivalent_aux (e : expr) :
  expr_equivalent e (lift_handler_conts e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [lift_handler_conts];
    try (intros vars conts statics hp; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (lift_handler_conts_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (lift_handler_conts_equivalent_aux body)).
  - assert (Hlayers :
      expr_equivalent
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler params metadata (lift_handler_conts handler))
           (lift_handler_conts body))).
    { apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (lift_handler_conts handler)) body)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_handler k params metadata Hole body) _ _
          (not_many _) (lift_handler_conts_equivalent_aux handler)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (lift_handler_conts handler)) Hole)
          _ _ (not_many _) (lift_handler_conts_equivalent_aux body)). }
    destruct
      (lift_conts k (Handler params metadata (lift_handler_conts handler))
         (lift_handler_conts body))
      as [lifted | ] eqn:Hlift; [ | exact Hlayers ].
    exact (expr_equivalent_trans _ _ _ Hlayers
      (lift_conts_equivalent k _ _ lifted Hlift)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (lift_handler_conts_equivalent_aux body)).
Qed.

Theorem lift_handler_conts_equivalent :
  lift_handler_conts_equivalent_statement.
Proof. exact lift_handler_conts_equivalent_aux. Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Theorem lift_handler_conts_equivariant :
  lift_handler_conts_equivariant_statement.
Proof. exact FlambdaLiftContsNamesProofs.lift_handler_conts_renamed. Qed.
