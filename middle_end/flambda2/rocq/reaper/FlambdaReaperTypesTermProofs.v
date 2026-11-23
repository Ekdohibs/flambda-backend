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

(* The proofs of FlambdaReaperTypesTermSpec.v. Nothing here needs review.

   The invariant of the run ([state_of_values]): the expression passes
   the check under the kinds of the values of the variables
   (FlambdaWellKindedSpec.kinds_of_env), so does every handler that a
   continuation keeps, under the kinds of the values it keeps
   ([cont_of_values]), and the static values are of the fragment. A step
   keeps it ([step_of_values]): a [Let] and a handler bind values of the
   kinds of their binders, which the machine tests
   (FlambdaWellKindedProofs.bind_params_kinds); a [Let_static] makes its
   blocks of the values of its fields, of kind value by the check
   ([eval_static_bindings_of_values]); what binds nothing static passes
   the check under any kinds ([unbound_of_values]), as the body of code
   that is called and the handlers of a recursive [Let_cont] do. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaHeapSpec FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellKindedSpec
  FlambdaWellKindedProofs FlambdaReaperTypesSpec FlambdaReaperTypesProofs
  FlambdaReaperTypesTermSpec.
Import ListNotations.

(* ======================================================================== *)
(* What binds nothing static                                                *)
(* ======================================================================== *)

Lemma unbound_of_values (e : expr) :
  ∀ kinds : kind_env, binds_statics e = false -> expr_of_values kinds e = true.
Proof.
  revert e. fix IH 1.
  intros [x kind d body | bindings body | k [params metadata handler] body
    | handlers body | ap | ac | scrutinee arms | ] kinds; simpl; intros Hb;
    try reflexivity.
  - exact (IH body _ Hb).
  - apply orb_false_iff in Hb as [Hbinds Hbody].
    rewrite (IH body _ Hbody), andb_true_r.
    apply forallb_forall. intros [id c | id | s const] Hin; try reflexivity.
    exfalso. apply (proj2 (not_true_iff_false _) Hbinds).
    apply existsb_exists. exists (Bind_symbol s const). split; [exact Hin |].
    reflexivity.
  - apply orb_false_iff in Hb as [Hh Hbody].
    rewrite (IH body _ Hbody), andb_true_r.
    destruct (bind_param_kinds params (hm_params_kinds metadata) kinds) as
      [handler_kinds | ]; [ | reflexivity ].
    exact (IH handler _ Hh).
  - apply orb_false_iff in Hb as [_ Hbody]. exact (IH body _ Hbody).
Qed.

Lemma handler_unbound_of_values (h : continuation_handler)
    (kinds : kind_env) :
  handler_binds_statics h = false -> handler_of_values kinds h = true.
Proof.
  destruct h as [params metadata handler]. simpl. intros Hh.
  destruct (bind_param_kinds params (hm_params_kinds metadata) kinds) as
    [handler_kinds | ]; [ | reflexivity ].
  exact (unbound_of_values handler _ Hh).
Qed.

(* ======================================================================== *)
(* The static values                                                        *)
(* ======================================================================== *)

Lemma statics_of_values_empty : statics_of_values empty_static_env.
Proof. intros s allocated Hl. discriminate Hl. Qed.

Lemma fields_of_values (vars : env) (fields : list simple) :
  ∀ vs : list value,
    forallb (fun s => has_kind (kind_of_simple (kinds_of_env vars) s) Value)
      fields = true ->
    eval_simples vars fields = Some vs ->
    ∀ v : value, v ∈ vs -> kind_of_value v = Value.
Proof.
  induction fields as [ | f fields IH]; simpl; intros vs Hall Heval.
  - injection Heval as <-. intros v [].
  - apply andb_true_iff in Hall as [Hf Hall].
    destruct (eval_simple vars f) as [v_f | ] eqn:Hv; [ | discriminate ].
    destruct (eval_simples vars fields) as [vs' | ] eqn:Hvs;
      [ | discriminate ].
    injection Heval as <-.
    intros v [<- | Hin]; [ | exact (IH vs' Hall eq_refl v Hin) ].
    apply has_kind_eq in Hf. rewrite (eval_simple_kind vars f v_f Hv) in Hf.
    congruence.
Qed.

(* A symbol bound to a constant that passes the check. *)
Lemma statics_of_values_bind (statics : static_env)
    (code : list (code_id * code)) (vars : env) (s : symbol)
    (const : static_const) (allocated : statically_allocated_value) :
  statics_of_values statics ->
  const_of_values (kinds_of_env vars) const = true ->
  eval_static_const vars const = Some allocated ->
  statics_of_values
    (Static_env code ((s, allocated) :: static_values statics)).
Proof.
  intros Hstatics Hconst Heval s' allocated'. simpl.
  destruct (Pos.eqb s' s); [ | exact (Hstatics s' allocated') ].
  intros [= <-].
  destruct const as [code_id value_slots | slot closure_symbols
    function_decls value_slots | tag fields | contents | bytes | elements];
    simpl in Heval, Hconst.
  - destruct (eval_value_slots vars value_slots); [ | discriminate ].
    injection Heval as <-. exact I.
  - destruct (lookup slot function_decls); [ | discriminate ].
    destruct (eval_value_slots vars value_slots); [ | discriminate ].
    injection Heval as <-. exact I.
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
      [ | discriminate ].
    injection Heval as <-. exact (fields_of_values vars fields vs Hconst Hvs).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hv; [ | discriminate ].
    injection Heval as <-.
    rewrite (eval_simple_kind vars contents v Hv) in Hconst.
    destruct (boxable_number_of_kind (kind_of_value v)); [ | discriminate ].
    discriminate.
  - injection Heval as <-. exact I.
  - destruct (eval_simples vars elements); [ | discriminate ].
    injection Heval as <-. exact I.
Qed.

Lemma eval_static_bindings_of_values (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    statics_of_values statics ->
    forallb (binding_of_values (kinds_of_env vars)) bindings = true ->
    eval_static_bindings vars bindings statics = Some statics' ->
    statics_of_values statics'.
Proof.
  induction bindings as [ | binding bindings IH]; simpl;
    intros statics statics' Hstatics Hall Heval.
  - injection Heval as <-. exact Hstatics.
  - apply andb_true_iff in Hall as [Hbinding Hall].
    destruct binding as [id c | id | s const]; simpl in Hbinding.
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      refine (IH _ _ _ Hall Heval). exact Hstatics.
    + exact (IH _ _ Hstatics Hall Heval).
    + destruct (lookup s (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
        [ | discriminate ].
      exact (IH _ _
               (statics_of_values_bind statics _ vars s const allocated
                  Hstatics Hbinding Hconst)
               Hall Heval).
Qed.

(* ======================================================================== *)
(* The invariant of the run                                                 *)
(* ======================================================================== *)

(* The handlers that a continuation keeps pass the check under the kinds
   of the values it keeps. *)
Fixpoint cont_of_values (c : cont_value) : bool :=
  match c with
  | Closure h vars conts recursive =>
    handler_of_values (kinds_of_env vars) h
    && forallb (fun '(_, h) => handler_of_values (kinds_of_env vars) h)
         recursive
    && forallb (fun '(_, c) => cont_of_values c) conts
  | Outer _ => true
  | Extra_args c _ => cont_of_values c
  | Over_application _ _ _ return_to exn_to =>
    cont_of_values return_to && cont_of_values exn_to
  end.

Definition conts_of_values (conts : cont_env) : bool :=
  forallb (fun '(_, c) => cont_of_values c) conts.

Definition state_of_values (s : state) : Prop :=
  expr_of_values (kinds_of_env (state_vars s)) (state_expr s) = true
  ∧ conts_of_values (state_conts s) = true
  ∧ statics_of_values (state_statics s).

Lemma resolve_of_values (k : continuation) (conts : cont_env) :
  conts_of_values conts = true -> cont_of_values (resolve k conts) = true.
Proof.
  unfold resolve, conts_of_values.
  induction conts as [ | [k' c] conts IH]; simpl; [ reflexivity | ].
  intros Hall. apply andb_true_iff in Hall as [Hc Hall].
  destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hall) ].
Qed.

Lemma with_extra_args_of_values (c : cont_value) (extra : list value) :
  cont_of_values c = true -> cont_of_values (with_extra_args c extra) = true.
Proof. destruct extra; simpl; intros Hc; exact Hc. Qed.

Lemma rec_closures_of_values
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  forallb (fun '(_, h) => handler_of_values (kinds_of_env vars) h) handlers
  = true ->
  conts_of_values conts = true ->
  conts_of_values (rec_closures handlers vars conts) = true.
Proof.
  intros Hall Hconts. unfold rec_closures, conts_of_values.
  apply forallb_forall. intros [k c] Hin.
  apply in_map_iff in Hin as [[k' h] [Heq Hin]]. injection Heq as <- <-.
  simpl. rewrite Hall. unfold conts_of_values in Hconts. rewrite Hconts.
  rewrite andb_true_r, andb_true_r.
  exact (proj1 (forallb_forall _ _) Hall (k', h) Hin).
Qed.

Lemma conts_of_values_app (conts conts' : cont_env) :
  conts_of_values conts = true -> conts_of_values conts' = true ->
  conts_of_values (conts ++ conts') = true.
Proof.
  unfold conts_of_values. rewrite forallb_app. intros -> ->. reflexivity.
Qed.

Lemma call_code_of_values (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  cont_of_values return_to = true -> cont_of_values exn_to = true ->
  statics_of_values statics ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp
    rd
  = Step s' ->
  state_of_values s'.
Proof.
  intros Hreturn Hexn Hstatics. unfold call_code.
  destruct (binds_statics (code_body c)) eqn:Hb; [ discriminate | ].
  match goal with
  | |- (if ?test then _ else _) = _ -> _ => destruct test
  end; [ discriminate | ].
  destruct (negb (values_of_subkinds _ _ _)); [ discriminate | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)]);
    [ | discriminate ].
  intros [= <-]. split; [ | split ]; simpl.
  - exact (unbound_of_values _ _ Hb).
  - rewrite Hreturn, Hexn. reflexivity.
  - exact Hstatics.
Qed.

Lemma run_call_of_values (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record) (s' : state) :
  (∀ (partial : value) (s'' : state),
     return_partial partial = Step s'' -> state_of_values s'') ->
  cont_of_values return_to = true -> cont_of_values exn_to = true ->
  statics_of_values statics ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  state_of_values s'.
Proof.
  intros Hpartial Hreturn Hexn Hstatics.
  destruct shape as [[closure c vs args_arity
    | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - exact (call_code_of_values _ _ _ _ _ _ _ _ _ _ _ Hreturn Hexn Hstatics).
  - apply (call_code_of_values _ _ _ _ _ _ _ _ _ _ _); [ | exact Hexn |
      exact Hstatics ].
    simpl. rewrite Hreturn, Hexn. reflexivity.
  - exact (Hpartial partial s').
  - discriminate.
Qed.

Lemma apply_cont_value_of_values (c : cont_value) :
  cont_of_values c = true ->
  ∀ (vs : list value) (statics : static_env) (hp : heap)
    (rd : read_record) (s' : state),
    statics_of_values statics ->
    apply_cont_value c vs statics hp rd = Step s' ->
    state_of_values s'.
Proof.
  induction c as [h def_vars def_conts recursive | k | c IH extra
    | remaining remaining_arity return_arity return_to IHreturn exn_to
      IHexn];
    simpl; intros Hc vs statics hp rd s' Hstatics.
  - destruct h as [params metadata handler]. simpl in Hc |- *.
    apply andb_true_iff in Hc as [Hc Hconts].
    apply andb_true_iff in Hc as [Hh Hrecursive].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Hkinds; [ | discriminate ].
    destruct (values_of_subkinds _ vs (hm_params_subkinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | discriminate ].
    intros [= <-]. split; [ | split ]; simpl.
    + apply kinds_eqb_eq in Hkinds. rewrite <- Hkinds in Hh.
      rewrite (bind_params_kinds params vs def_vars handler_vars Hbind) in Hh.
      exact Hh.
    + apply conts_of_values_app; [ | exact Hconts ].
      exact (rec_closures_of_values recursive def_vars def_conts Hrecursive
               Hconts).
    + exact Hstatics.
  - discriminate.
  - exact (IH Hc _ _ _ _ _ Hstatics).
  - apply andb_true_iff in Hc as [Hreturn Hexn].
    destruct vs as [ | v [ | ? ?]]; try discriminate.
    apply (run_call_of_values _ _ _ _ _ _ _ _ _); try assumption.
    intros partial s'' Hs''.
    exact (IHreturn Hreturn _ _ _ _ _ Hstatics Hs'').
Qed.

Lemma step_of_values (s s' : state) :
  state_of_values s -> step s = Step s' -> state_of_values s'.
Proof.
  destruct s as [e vars conts statics hp rd].
  intros [He [Hconts Hstatics]]. simpl in He, Hconts, Hstatics.
  destruct e as [x kind d body | bindings body | k h body | handlers body
    | ap | ac | scrutinee arms | ]; simpl in He; cbn [step].
  - destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | discriminate ].
    intros [= <-]. split; [ | split ]; simpl; [ | exact Hconts |
      exact Hstatics ].
    apply kind_eqb_eq in Hkind.
    change (kinds_of_env ((x, v) :: vars))
      with ((x, kind_of_value v) :: kinds_of_env vars).
    rewrite Hkind. exact He.
  - apply andb_true_iff in He as [Hbindings Hbody].
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    intros [= <-]. split; [ exact Hbody | split; [ exact Hconts | ] ].
    exact (eval_static_bindings_of_values vars bindings statics statics'
             Hstatics Hbindings Heval).
  - apply andb_true_iff in He as [Hh Hbody].
    intros [= <-]. split; [ exact Hbody | split; [ | exact Hstatics ] ].
    unfold conts_of_values in *. simpl. rewrite Hh, Hconts. reflexivity.
  - destruct (handlers_bind_statics handlers) eqn:Hhb; [ discriminate | ].
    intros [= <-]. split; [ exact He | split; [ | exact Hstatics ] ].
    apply conts_of_values_app; [ | exact Hconts ].
    apply rec_closures_of_values; [ | exact Hconts ].
    apply forallb_forall. intros [k h] Hin.
    apply handler_unbound_of_values.
    destruct (handler_binds_statics h) eqn:Hh; [ | reflexivity ].
    exfalso. apply (proj2 (not_true_iff_false _) Hhb).
    apply existsb_exists. exists (k, h). split; [ exact Hin | exact Hh ].
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    unfold call_function.
    apply (run_call_of_values _ _ _ _ _ _ _ _ _).
    + intros partial s'' Hs''.
      exact (apply_cont_value_of_values _
               (resolve_of_values _ _ Hconts) _ _ _ _ _ Hstatics Hs'').
    + exact (resolve_of_values _ _ Hconts).
    + exact (with_extra_args_of_values _ _ (resolve_of_values _ _ Hconts)).
    + exact Hstatics.
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    exact (apply_cont_value_of_values _ (resolve_of_values _ _ Hconts) _ _
             _ _ _ Hstatics).
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. split; [ reflexivity | split; assumption ].
  - discriminate.
Qed.

Lemma run_of_values (n : nat) :
  ∀ s s' : state, state_of_values s -> run n s = Step s' -> state_of_values s'.
Proof.
  induction n as [ | n IH]; simpl; intros s s' Hs.
  - intros [= <-]. exact Hs.
  - destruct (step s) as [s'' | | ] eqn:Hstep; try discriminate.
    exact (IH s'' s' (step_of_values s s'' Hs Hstep)).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem unit_of_values_runs : unit_of_values_runs_statement.
Proof.
  intros e n s He Hrun.
  assert (Hinit :
            state_of_values
              (State e [] [] empty_static_env empty_heap no_reads)).
  { split; [ exact He | split; [ reflexivity | ] ].
    exact statics_of_values_empty. }
  exact (proj2 (proj2 (run_of_values n _ _ Hinit Hrun))).
Qed.

Theorem unit_of_values_exits : unit_of_values_exits_statement.
Proof.
  intros e He n s k vs hp rd Hrun _. exact (unit_of_values_runs e n s He Hrun).
Qed.

Lemma well_kinded_of_values (slots : slot_kinds) (e : expr) :
  ∀ (kinds : kind_env) (conts : arity_env),
    well_kinded slots kinds conts e = true -> expr_of_values kinds e = true.
Proof.
  revert e. fix IH 1.
  intros [x kind d body | bindings body | k [params metadata handler] body
    | handlers body | ap | ac | scrutinee arms | ] kinds conts; simpl;
    intros Hwk; try reflexivity.
  - apply andb_true_iff in Hwk as [_ Hbody]. exact (IH body _ _ Hbody).
  - apply andb_true_iff in Hwk as [Hbindings Hbody].
    rewrite (IH body _ _ Hbody), andb_true_r.
    apply forallb_forall. intros binding Hin.
    pose proof (proj1 (forallb_forall _ _) Hbindings binding Hin) as Hb.
    destruct binding as [id c | id | s const]; simpl in Hb |- *;
      try reflexivity.
    destruct const; simpl in Hb |- *; first [ reflexivity | exact Hb ].
  - apply andb_true_iff in Hwk as [Hh Hbody].
    rewrite (IH body _ _ Hbody), andb_true_r.
    apply andb_true_iff in Hh as [_ Hh].
    destruct (bind_param_kinds params (hm_params_kinds metadata) kinds) as
      [handler_kinds | ]; [ | reflexivity ].
    exact (IH handler _ _ Hh).
  - apply andb_true_iff in Hwk as [_ Hbody]. exact (IH body _ _ Hbody).
Qed.

Theorem well_kinded_unit_of_values : well_kinded_unit_of_values_statement.
Proof.
  intros slots unit_return unit_exn e Hwk.
  exact (well_kinded_of_values slots e _ _ Hwk).
Qed.

Theorem naked_field_not_of_values : naked_field_not_of_values_statement.
Proof. reflexivity. Qed.

(* The statements of FlambdaReaperTypesSpec.v, with [unit_of_values]. *)

Theorem exit_relation_term : exit_relation_term_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact Honce Hvals.
  exact (rebuild_exit_relation_holds flag is_local u e T b Hlocal Hslots Hfrag
           Hdc Hrk Hwf Hexits Hcheck Hcheck_exact Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem exit_relation_whole_term : exit_relation_whole_term_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact Honce Hvals.
  exact (rebuild_exit_relation_whole_holds flag is_local u e T b Hlocal Hslots
           Hfrag Hdc Hrk Hwf Hexits Hcheck Hcheck_exact Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem types_end_to_end_term : types_end_to_end_term_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals.
  exact (exported_types_sound_end_to_end flag is_local u e T b sets tenv canon
           valuation valuation' n s k vs hp rd sym allocated Hlocal Hslots
           Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem types_end_to_end_term_landed : types_end_to_end_term_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals.
  exact (exported_types_sound_end_to_end_landed flag is_local u e T b sets
           tenv canon valuation valuation' n s k vs hp rd sym allocated Hlocal
           Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem unaliased_types_end_to_end_term :
  unaliased_types_end_to_end_term_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals.
  exact (exported_unaliased_types_sound_end_to_end flag is_local u e T b sets
           tenv canon valuation valuation' n s k vs hp rd sym allocated Hlocal
           Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem unaliased_types_end_to_end_term_landed :
  unaliased_types_end_to_end_term_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals.
  exact (exported_unaliased_types_sound_end_to_end_landed flag is_local u e T
           b sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem exit_relation_term_local : exit_relation_term_local_statement.
Proof.
  intros flag is_local u e T b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
    Hcheck_exact Honce Hvals.
  exact (rebuild_exit_relation_holds_local flag is_local u e T b Hlocal Hfrag
           Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem exit_relation_whole_term_local :
  exit_relation_whole_term_local_statement.
Proof.
  intros flag is_local u e T b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
    Hcheck_exact Honce Hvals.
  exact (rebuild_exit_relation_whole_holds_local flag is_local u e T b Hlocal
           Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem types_end_to_end_term_local : types_end_to_end_term_local_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals.
  exact (exported_types_sound_end_to_end_local flag is_local u e T b sets tenv
           canon valuation valuation' n s k vs hp rd sym allocated Hlocal Hfrag
           Hdc Hrk Hwf Hexits Hleast Honce (unit_of_values_exits e Hvals)).
Qed.

Theorem types_end_to_end_term_local_landed :
  types_end_to_end_term_local_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals.
  exact (exported_types_sound_end_to_end_local_landed flag is_local u e T b
           sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem unaliased_types_end_to_end_term_local :
  unaliased_types_end_to_end_term_local_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals.
  exact (exported_unaliased_types_sound_end_to_end_local flag is_local u e T b
           sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (unit_of_values_exits e Hvals)).
Qed.

Theorem unaliased_types_end_to_end_term_local_landed :
  unaliased_types_end_to_end_term_local_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals.
  exact (exported_unaliased_types_end_to_end_local_landed flag is_local u e T b
           sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (unit_of_values_exits e Hvals)).
Qed.
